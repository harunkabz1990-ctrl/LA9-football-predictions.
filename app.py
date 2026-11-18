from __future__ import annotations
import csv, io, json, math, os, sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT=Path(__file__).parent
DB_PATH=Path(os.getenv('LA9_DB_PATH', ROOT/'la9.sqlite3'))
API_KEY=os.getenv('APISPORTS_KEY','').strip()
API_BASE='https://v3.football.api-sports.io'
LEAGUES={
 'EPL':39,'English Championship':40,'La Liga':140,'Bundesliga':78,
 'Serie A':135,'Ligue 1':61,'Dutch Eredivisie':88,'Portuguese Primeira Liga':94,
 'Kenyan Premier League':276
}
app=FastAPI(title='LA-9 Football Intelligence',version='2.0.0')

def db():
 c=sqlite3.connect(DB_PATH); c.row_factory=sqlite3.Row; return c

def init_db():
 with db() as c:
  c.executescript('''
  CREATE TABLE IF NOT EXISTS matches(id INTEGER PRIMARY KEY, fixture_id TEXT UNIQUE, date TEXT NOT NULL, competition TEXT NOT NULL, season TEXT, home_team TEXT NOT NULL, away_team TEXT NOT NULL, home_goals INTEGER, away_goals INTEGER, status TEXT DEFAULT 'FT', source TEXT DEFAULT 'CSV');
  CREATE INDEX IF NOT EXISTS ix_matches_lookup ON matches(competition,date);
  CREATE UNIQUE INDEX IF NOT EXISTS ux_match_identity ON matches(competition,date,home_team,away_team);
  CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY, fixture_key TEXT UNIQUE, date TEXT, competition TEXT, home_team TEXT, away_team TEXT, p_home REAL, p_draw REAL, p_away REAL, pick TEXT, confidence REAL, result TEXT DEFAULT 'PENDING', created_at TEXT);
  CREATE TABLE IF NOT EXISTS sync_log(id INTEGER PRIMARY KEY, at TEXT, task TEXT, status TEXT, detail TEXT);
  ''')
init_db()

class FixtureIn(BaseModel):
 date:str; competition:str; home_team:str; away_team:str

class SettleIn(BaseModel): result:str

def log(task,status,detail):
 with db() as c: c.execute('INSERT INTO sync_log(at,task,status,detail) VALUES(?,?,?,?)',(datetime.now(timezone.utc).isoformat(),task,status,detail[:1000]))

def parse_dt(s):
 try: return datetime.fromisoformat(str(s).replace('Z','+00:00'))
 except: return datetime.strptime(str(s)[:10],'%Y-%m-%d').replace(tzinfo=timezone.utc)

def get_history(comp, cutoff, team=None):
 with db() as c:
  if team:
   rows=c.execute('SELECT * FROM matches WHERE competition=? AND date<? AND (home_team=? OR away_team=?) AND home_goals IS NOT NULL ORDER BY date DESC LIMIT 10',(comp,cutoff,team,team)).fetchall()
  else: rows=c.execute('SELECT * FROM matches WHERE competition=? AND date<? AND home_goals IS NOT NULL ORDER BY date DESC',(comp,cutoff)).fetchall()
 return [dict(x) for x in rows]

def stats(rows,team,venue=None):
 rr=[r for r in rows if (r['home_team']==team if venue=='home' else r['away_team']==team if venue=='away' else True)]
 if not rr:return None
 gf=[];ga=[];pts=[]
 for r in rr:
  home=r['home_team']==team; x=r['home_goals'] if home else r['away_goals']; y=r['away_goals'] if home else r['home_goals']
  gf.append(x);ga.append(y);pts.append(3 if x>y else 1 if x==y else 0)
 return {'gf':sum(gf)/len(gf),'ga':sum(ga)/len(ga),'pts':sum(pts)/len(pts),'n':len(rr)}

def poisson(k,l): return math.exp(-l)*l**k/math.factorial(k)

def predict(f):
 comp=f['competition']; kickoff=parse_dt(f['date']).astimezone(timezone.utc).isoformat()
 hrows=get_history(comp,kickoff,f['home_team']); arows=get_history(comp,kickoff,f['away_team'])
 h=stats(hrows,f['home_team']); a=stats(arows,f['away_team'])
 if not h or not a or h['n']<3 or a['n']<3:
  return {**f,'status':'INSUFFICIENT_HISTORY','message':'Need at least 3 earlier completed matches for each team in this competition.'}
 hh=stats(hrows,f['home_team'],'home'); aa=stats(arows,f['away_team'],'away')
 venue=(hh['pts']-aa['pts']) if hh and aa else 0
 hidden=h['gf']-a['ga']; opponent=(h['gf']+a['ga'])-(a['gf']+h['ga'])
 # Nine transparent signals. Missing inputs are neutral, never invented.
 fatigue=0.0; motivation=0.0; market=0.0; chaos=0.0
 signal_values=[hidden,venue,-(h['gf']-h['ga']-a['gf']+a['ga']),opponent,motivation,fatigue,market,chaos,1 if hidden*venue>0 else -1 if hidden*venue<0 else 0]
 weights=[.18,.16,.12,.12,.05,.08,.09,.10,.10]
 scales=[1.5,1.1,1.4,1.5,1,1,1,1,1]
 score=sum(math.tanh(v/s)*w for v,w,s in zip(signal_values,weights,scales))
 home_adv=.28+score*1.8
 lh=max(.25,min(3.5,(h['gf']+a['ga'])/2*math.exp(home_adv*.25)))
 la=max(.20,min(3.3,(a['gf']+h['ga'])/2*math.exp(-home_adv*.25)))
 ph=pd=pa=0
 for i in range(9):
  for j in range(9):
   p=poisson(i,lh)*poisson(j,la)
   if i>j:ph+=p
   elif i==j:pd+=p
   else:pa+=p
 total=ph+pd+pa; ph,pd,pa=ph/total,pd/total,pa/total
 probs={'HOME':ph,'DRAW':pd,'AWAY':pa}; pick=max(probs,key=probs.get); confidence=probs[pick]
 # Conservative flag; no promise of winning or profitability.
 decision=pick if confidence>=.52 else 'NO BET'
 return {**f,'status':'OK','p_home':round(ph,4),'p_draw':round(pd,4),'p_away':round(pa,4),'pick':decision,'most_likely':pick,'confidence':round(confidence,4),'expected_goals':f'{lh:.2f}-{la:.2f}','sample_home':h['n'],'sample_away':a['n'],'engines':{k:round(v,3) for k,v in zip(['Hidden strength','Venue DNA','Fortune decomposition','Opponent weakness','Motivation/context','Fatigue','Market blind spot','Chaos detector','Signal agreement'],signal_values)}}

@app.get('/',response_class=HTMLResponse)
def home(): return (ROOT/'index.html').read_text(encoding='utf-8')
@app.get('/api/status')
def status():
 with db() as c:
  n=c.execute('select count(*) from matches where home_goals is not null').fetchone()[0]
  teams=c.execute('select count(distinct competition||"|"||team) from (select competition,home_team team from matches union select competition,away_team team from matches)').fetchone()[0]
  p=c.execute('select count(*) from predictions').fetchone()[0]; settled=c.execute("select count(*) from predictions where result!='PENDING'").fetchone()[0]
  logs=[dict(x) for x in c.execute('select * from sync_log order by id desc limit 5').fetchall()]
 return {'historical_matches':n,'teams':teams,'predictions':p,'settled':settled,'provider_configured':bool(API_KEY),'provider':'API-Sports / API-Football','leagues':LEAGUES,'recent_sync':logs}

@app.post('/api/history/import')
async def import_history(file:UploadFile=File(...)):
 raw=(await file.read()).decode('utf-8-sig')
 try: rows=list(csv.DictReader(io.StringIO(raw)))
 except Exception as e: raise HTTPException(400,str(e))
 required=['date','competition','season','home_team','away_team','home_goals','away_goals']
 if not rows or any(k not in rows[0] for k in required): raise HTTPException(400,'CSV must include: '+', '.join(required))
 count=0; errors=[]
 with db() as c:
  for i,r in enumerate(rows,2):
   try:
    dt=parse_dt(r['date']).astimezone(timezone.utc).isoformat()
    hg=int(r['home_goals']); ag=int(r['away_goals'])
    if hg<0 or ag<0: raise ValueError('negative score')
    c.execute('INSERT OR IGNORE INTO matches(date,competition,season,home_team,away_team,home_goals,away_goals,status,source) VALUES(?,?,?,?,?,?,?,?,?)',(dt,r['competition'].strip(),r.get('season',''),r['home_team'].strip(),r['away_team'].strip(),hg,ag,'FT','CSV'))
    count+=c.execute('SELECT changes()').fetchone()[0]
   except Exception as e: errors.append(f'row {i}: {e}')
 log('historical CSV import','OK',f'{count} rows accepted; {len(errors)} errors')
 return {'imported':count,'errors':errors[:20]}

@app.post('/api/fixtures/manual')
def add_fixture(f:FixtureIn):
 with db() as c:
  c.execute('INSERT OR IGNORE INTO matches(fixture_id,date,competition,home_team,away_team,status,source) VALUES(?,?,?,?,?,?,?)',('manual:'+f.date+f.competition+f.home_team+f.away_team,parse_dt(f.date).isoformat(),f.competition,f.home_team,f.away_team,'NS','Manual'))
 return {'ok':True}

@app.get('/api/fixtures')
def fixtures(days:int=10):
 start=datetime.now(timezone.utc).isoformat(); end=(datetime.now(timezone.utc)+timedelta(days=max(1,min(days,30)))).isoformat()
 with db() as c: rows=c.execute("select * from matches where status in ('NS','TBD') and date>=? and date<=? order by date",(start,end)).fetchall()
 return [dict(x) for x in rows]

@app.post('/api/sync/fixtures')
def sync_fixtures(days:int=10):
 if not API_KEY: raise HTTPException(400,'Add APISPORTS_KEY to the server environment before syncing live fixtures.')
 start=date.today().isoformat(); end=(date.today()+timedelta(days=max(1,min(days,14)))).isoformat(); added=0; errs=[]
 for name,lid in LEAGUES.items():
  try:
   res=requests.get(API_BASE+'/fixtures',headers={'x-apisports-key':API_KEY},params={'league':lid,'season':date.today().year,'from':start,'to':end},timeout=25)
   res.raise_for_status(); payload=res.json()
   if payload.get('errors'): raise RuntimeError(str(payload['errors']))
   for item in payload.get('response',[]):
    fx=item['fixture']; teams=item['teams']; league=item['league']
    with db() as c:
     c.execute('INSERT OR IGNORE INTO matches(fixture_id,date,competition,season,home_team,away_team,status,source) VALUES(?,?,?,?,?,?,?,?)',(str(fx['id']),fx['date'],name,str(league.get('season','')),teams['home']['name'],teams['away']['name'],fx['status']['short'],'API-Sports'))
     added+=c.execute('select changes()').fetchone()[0]
  except Exception as e: errs.append(f'{name}: {e}')
 log('live fixture sync','OK' if not errs else 'PARTIAL',f'{added} fixtures; '+ '; '.join(errs))
 return {'fixtures_added':added,'errors':errs,'from':start,'to':end}

@app.post('/api/sync/results')
def sync_results(days:int=5):
 if not API_KEY: raise HTTPException(400,'Add APISPORTS_KEY to the server environment before syncing results.')
 start=(date.today()-timedelta(days=max(1,min(days,14)))).isoformat(); end=date.today().isoformat(); updated=0; errs=[]
 for name,lid in LEAGUES.items():
  try:
   res=requests.get(API_BASE+'/fixtures',headers={'x-apisports-key':API_KEY},params={'league':lid,'season':date.today().year,'from':start,'to':end},timeout=25);res.raise_for_status(); payload=res.json()
   if payload.get('errors'): raise RuntimeError(str(payload['errors']))
   for item in payload.get('response',[]):
    fx=item['fixture']; teams=item['teams']; goals=item['goals']; status=fx['status']['short']
    if status in ('FT','AET','PEN') and goals['home'] is not None:
     with db() as c:
      c.execute('UPDATE matches SET home_goals=?,away_goals=?,status=? WHERE fixture_id=?',(goals['home'],goals['away'],status,str(fx['id'])));updated+=c.execute('select changes()').fetchone()[0]
  except Exception as e: errs.append(f'{name}: {e}')
 log('result sync','OK' if not errs else 'PARTIAL',f'{updated} results; '+ '; '.join(errs))
 return {'results_updated':updated,'errors':errs}

@app.post('/api/history/sync')
def sync_history():
    if not API_KEY:
        raise HTTPException(400, 'Add APISPORTS_KEY to the server environment.')

    added = 0
    errors = []
    seasons = [2022, 2023, 2024]

    for name, lid in LEAGUES.items():
        for season in seasons:
            try:
                res = requests.get(
                    API_BASE + '/fixtures',
                    headers={'x-apisports-key': API_KEY},
                    params={'league': lid, 'season': season},
                    timeout=30
                )
                res.raise_for_status()
                payload = res.json()

                if payload.get('errors'):
                    raise RuntimeError(str(payload['errors']))

                for item in payload.get('response', []):
                    fx = item['fixture']
                    teams = item['teams']
                    goals = item['goals']
                    status = fx['status']['short']

                    if status not in ('FT', 'AET', 'PEN'):
                        continue

                    if goals['home'] is None or goals['away'] is None:
                        continue

                    with db() as c:
                        c.execute(
                            '''INSERT OR IGNORE INTO matches
                            (fixture_id, date, competition, season,
                             home_team, away_team, home_goals, away_goals,
                             status, source)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                            (
                                str(fx['id']),
                                fx['date'],
                                name,
                                str(season),
                                teams['home']['name'],
                                teams['away']['name'],
                                goals['home'],
                                goals['away'],
                                status,
                                'API-Sports Historical'
                            )
                        )
                        added += c.execute(
                            'SELECT changes()'
                        ).fetchone()[0]

            except Exception as e:
                errors.append(f'{name} {season}: {e}')

    log(
        'historical API sync',
        'OK' if not errors else 'PARTIAL',
        f'{added} historical matches; ' + '; '.join(errors)
    )

    return {
        'matches_added': added,
        'errors': errors,
        'seasons': seasons
    }

@app.post('/api/predict')
def run_predictions():
 with db() as c: rows=[dict(x) for x in c.execute("select * from matches where status in ('NS','TBD') and date>=? order by date",(datetime.now(timezone.utc).isoformat(),)).fetchall()]
 results=[]
 for f in rows:
  p=predict(f);results.append(p)
  if p.get('status')=='OK':
   key='|'.join([p['date'],p['competition'],p['home_team'],p['away_team']])
   with db() as c: c.execute('INSERT OR IGNORE INTO predictions(fixture_key,date,competition,home_team,away_team,p_home,p_draw,p_away,pick,confidence,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)',(key,p['date'],p['competition'],p['home_team'],p['away_team'],p['p_home'],p['p_draw'],p['p_away'],p['pick'],p['confidence'],datetime.now(timezone.utc).isoformat()))
 return {'count':len(results),'predictions':results}

@app.get('/api/predictions')
def get_predictions():
 with db() as c: return [dict(x) for x in c.execute('select * from predictions order by date desc').fetchall()]
@app.post('/api/predictions/{pid}/settle')
def settle(pid:int,data:SettleIn):
 if data.result not in ('HOME','DRAW','AWAY'): raise HTTPException(400,'Result must be HOME, DRAW or AWAY')
 with db() as c:
  cur=c.execute('update predictions set result=? where id=? and result="PENDING"',(data.result,pid))
  if not cur.rowcount: raise HTTPException(404,'Prediction not found or already settled')
 return {'ok':True}
@app.get('/api/health')
def health(): return {'ok':True,'version':'2.0.0'}

# Optional scheduler: hourly fixtures + daily result refresh. Requires API key.
try:
 from apscheduler.schedulers.background import BackgroundScheduler
 scheduler=BackgroundScheduler(timezone='UTC')
 if API_KEY:
  scheduler.add_job(lambda: _scheduled('/api/sync/fixtures'), 'interval', hours=6, id='fixtures', replace_existing=True)
  scheduler.add_job(lambda: _scheduled('/api/sync/results'), 'interval', hours=6, id='results', replace_existing=True)
 scheduler.start()
except Exception: pass

def _scheduled(path):
 try:
  # Invoke internal sync logic over local HTTP once server is running.
  requests.post('http://127.0.0.1:'+os.getenv('PORT','8000')+path,timeout=180)
 except Exception as e: log(path,'ERROR',str(e))
