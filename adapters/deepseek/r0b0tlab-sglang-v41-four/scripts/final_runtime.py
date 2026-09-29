"""Pinned TP4 launch/stop. No model traffic; new run roots and exact-ID ownership."""
from __future__ import annotations
import argparse,concurrent.futures,hashlib,json,os,pathlib,shlex,subprocess,sys,time,urllib.request

ROOT=pathlib.Path(__file__).resolve().parents[1]
HOME=pathlib.Path.home()
from cluster_config import CONFIG, require as require_cluster
CFG=CONFIG or {}
if CONFIG is not None:os.environ['DSV41_SSH_KNOWN_HOSTS']=CONFIG['ssh_known_hosts_file']
HOSTS=CFG.get('ssh_hosts',[None]*4)
NAMES=CFG.get('hostnames',[None]*4)
FABRIC_IPS=CFG.get('fabric_ips',[None]*4)
DIST_INIT_ADDR=CFG.get('dist_init_addr','CONFIGURE_INVENTORY')
SSH=['ssh','-i',CFG.get('ssh_identity_file','CONFIGURE_INVENTORY'),'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+CFG.get('ssh_known_hosts_file','CONFIGURE_INVENTORY'),'-o','ConnectTimeout=8']
MODEL=pathlib.Path(CFG.get('model_path','CONFIGURE_INVENTORY'))


def run(argv,timeout=30):
    p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
    if p.returncode:raise RuntimeError(f'command failed rc={p.returncode}: {shlex.join(argv[:5])}: {p.stderr[-1500:]}')
    return p.stdout


def remote(rank,argv,timeout=30):
    require_cluster()
    return run([*SSH,HOSTS[rank],shlex.join(argv)],timeout)


def save(p,obj):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def get(endpoint):
    for attempt in range(2):
        try:return json.load(urllib.request.urlopen('http://127.0.0.1:30000/'+endpoint,timeout=30))
        except (TimeoutError,urllib.error.URLError):
            if attempt:raise
            time.sleep(1)


def idle(loads,dp=1):
    return (isinstance(loads,list) and len(loads)==dp and {x.get('dp_rank') for x in loads}==set(range(dp))
            and all(type(x.get('num_reqs')) is int and x['num_reqs']==0 and type(x.get('num_waiting_reqs')) is int and x['num_waiting_reqs']==0 for x in loads))


def profile(name):
    if name not in ('prod','prod-k3','prod-c8','1m'):raise ValueError('unknown final profile')
    p=ROOT/'profiles'/f'dsv41-{name}.env'
    data=run(['bash','-c','set -a; source "$1"; env -0','profile',str(p)])
    d=dict(x.split('=',1) for x in data.split('\0') if '=' in x)
    for k in ('TP_SIZE','EP_SIZE','NNODES'):assert int(d[k])==4
    assert d['MEM_FRACTION_STATIC']=='0.80' and d['CHUNKED_PREFILL_SIZE'] in ('256','512')
    assert d['DSPARK_BLOCK_SIZE'] in ('3','5')
    assert int(d['CONTEXT_LENGTH']) in (524288,1048576)
    assert int(d['MAX_RUNNING_REQUESTS']) in (1,8)
    assert d['DSV41_ADAPTIVE_CHUNK']=='1'
    assert d['PYTORCH_CUDA_ALLOC_CONF'] in ('expandable_segments:False','expandable_segments:False,garbage_collection_threshold:0.6')
    if name=='prod':
        assert d['MAX_RUNNING_REQUESTS']=='1'
        assert d['CHUNKED_PREFILL_SIZE']=='256'
        assert d['CONTEXT_LENGTH']=='524288'
        assert d['DSV41_CHUNK_MAX']=='256'
        assert d['DSV41_INDEXER_K_CHUNK_MAX']=='2048'
        assert d['DSV41_TORCH_INDEXER_KSLICE']=='1'
    if name=='prod-c8':
        assert d['MAX_RUNNING_REQUESTS']=='8'
        assert 'garbage_collection_threshold' not in d['PYTORCH_CUDA_ALLOC_CONF']
    return d,hashlib.sha256(p.read_bytes()).hexdigest()


def make_argv(rank,p,image,labels,iface,nccl):
    if not image.startswith('sha256:') or len(image)!=71 or any(c not in '0123456789abcdef' for c in image[7:]):raise ValueError('immutable image config ID required')
    env={'DSV41_ENGRAM_FILE_STORE':'1','DSV41_MODEL_PATH':'/model','DSV41_ADAPTIVE_CHUNK':'1',
         'DSV41_CHUNK_BUDGET_TOKENS2':p['DSV41_CHUNK_BUDGET_TOKENS2'],'DSV41_CHUNK_MIN':p.get('DSV41_CHUNK_MIN','256'),
         'DSV41_CHUNK_MAX':p['DSV41_CHUNK_MAX'],
         'DSV41_INDEXER_SCORE_BUDGET_MIB':p['DSV41_INDEXER_SCORE_BUDGET_MIB'],
         'DSV41_INDEXER_K_CHUNK_MAX':p.get('DSV41_INDEXER_K_CHUNK_MAX','2048'),
         'DSV41_TORCH_INDEXER_KSLICE':p.get('DSV41_TORCH_INDEXER_KSLICE','0'),
         'SGLANG_RAGGED_VERIFY_MODE':'static','SGLANG_SIMULATE_ACC_LEN':'-1',
         'SGLANG_DSPARK_ENABLE_SPS_RECORD':'0','SGLANG_VIT_ENABLE_CUDA_GRAPH':'0','SGLANG_FLASHINFER_MOE_FUSED_FINALIZE':'0',
         'PYTORCH_CUDA_ALLOC_CONF':p['PYTORCH_CUDA_ALLOC_CONF'],'SGLANG_SM120_FLASHMLA_BACKEND':'flashinfer',
         **nccl,'NCCL_BUFFSIZE':p.get('NCCL_BUFFSIZE','4194304'),'GLOO_SOCKET_IFNAME':iface,'NCCL_SOCKET_IFNAME':iface,'PORT':'30000'}
    a=['docker','run','-d','--name','dsv41-rank','--network','host','--ipc','host','--runtime','nvidia','--device','/dev/infiniband','--cap-add','CAP_IPC_LOCK','--ulimit','memlock=-1','-v',str(MODEL)+':/model:ro']
    for k,v in labels.items():a+=['--label',k+'='+v]
    for k,v in env.items():a+=['-e',k+'='+v]
    a+=[image,'python3','-m','sglang.launch_server','--model-path','/model','--trust-remote-code','--tp-size','4','--ep-size','4','--nnodes','4','--node-rank',str(rank),'--dist-init-addr',DIST_INIT_ADDR,'--host','0.0.0.0','--port','30000']
    for key,flag in [('CONTEXT_LENGTH','--context-length'),('MAX_TOTAL_TOKENS','--max-total-tokens'),('MAX_RUNNING_REQUESTS','--max-running-requests'),('CHUNKED_PREFILL_SIZE','--chunked-prefill-size'),('MEM_FRACTION_STATIC','--mem-fraction-static'),('DSPARK_BLOCK_SIZE','--speculative-dspark-block-size')]:a += [flag,p[key]]
    a+=['--speculative-algorithm','DSPARK','--cuda-graph-max-bs-decode',p['MAX_RUNNING_REQUESTS'],'--prefill-max-requests','1','--min-free-slots-delay','1','--weight-loader-drop-cache-after-load','--reasoning-parser','auto','--tool-call-parser','auto','--fp8-gemm-backend','cutlass','--watchdog-timeout','3600','--enable-metrics']
    return a


def labels_for(rank,epoch,source,phash,image):
    overlay_v2='sha256:5b246919f183289ab2a147f0ba7f22c54a72c52748082ce4cd2ec8303d38fa1f'
    candidate='dsv41-tp4-sm121-overlay-v2' if image==overlay_v2 else 'dsv41-tp4-sm121-overlay-v4'
    nonce=hashlib.sha256('\0'.join([candidate,source,epoch,str(rank),phash,image]).encode()).hexdigest()
    return {'org.r0b0tlab.'+k:v for k,v in dict(candidate_id=candidate,candidate_source_sha=source,profile_sha256=phash,epoch=epoch,rank=str(rank),owner_nonce=nonce,image_id=image).items()}


def preflight(rank,image):
    assert remote(rank,['hostname']).strip()==NAMES[rank]
    d=json.loads(remote(rank,['docker','image','inspect',image]))[0];assert d['Id']==image
    assert not remote(rank,['docker','ps','-q']).strip(),'active containers; refusing launch'
    probe="import json,pathlib,hashlib,sys; m=dict(l.split(':',1) for l in pathlib.Path('/proc/meminfo').read_text().splitlines()); print(json.dumps({'available':int(m['MemAvailable'].split()[0])*1024,'index':hashlib.sha256((pathlib.Path(sys.argv[1])/'model.safetensors.index.json').read_bytes()).hexdigest()}))"
    mem=json.loads(remote(rank,['python3','-c',probe,str(MODEL)]));assert mem['available']>=24*2**30,'24 GiB launch reserve not met'
    ips=json.loads(remote(rank,['ip','-j','addr']))
    iface=next(x['ifname'] for x in ips if any(z.get('local')==FABRIC_IPS[rank] for z in x.get('addr_info',[])))
    return dict(rank=rank,image_id=image,iface=iface,**mem)


def launch(args):
    cfg=require_cluster()
    p,phash=profile(args.profile)
    image=args.image_id
    source=run(['git','-C',str(ROOT),'rev-parse','HEAD']).strip()
    if not args.dry_run and run(['git','-C',str(ROOT),'status','--porcelain','--','adapter','docker','scripts','profiles']).strip():raise RuntimeError('dirty launch inputs')
    epoch='dsv41final-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+args.profile
    ncclpath=pathlib.Path(cfg['nccl_env'])
    raw=run(['bash','-c','set -a; source "$1"; env -0','nccl',str(ncclpath)])
    nccl={k:v for k,v in (x.split('=',1) for x in raw.split('\0') if '=' in x) if k.startswith('NCCL_') and k!='NCCL_SOCKET_IFNAME'}
    assert nccl.get('NCCL_NET')=='IB' and nccl.get('NCCL_IB_DISABLE')=='0'
    if args.dry_run:
        print(json.dumps([make_argv(r,p,image,labels_for(r,epoch,source,phash,image),'enp1s0f0np0',nccl) for r in range(4)],indent=2));return
    root=pathlib.Path(args.run_dir).resolve();root.mkdir(parents=True,exist_ok=False)
    record=dict(epoch=epoch,source_sha=source,profile=args.profile,profile_sha=phash,image_id=image,base_url='http://127.0.0.1:30000',ranks=[])
    save(root/'launch.json',record)
    with concurrent.futures.ThreadPoolExecutor(4) as ex:checks=list(ex.map(lambda r:preflight(r,image),range(4)))
    assert len({x['index'] for x in checks})==1,'checkpoint index mismatch'
    save(root/'preflight.json',checks)
    for r in (1,2,3,0):
        # Cache reclaim only with no active serving container, after admission.
        remote(r,['sudo','-n','sh','-c','sync; printf 3 > /proc/sys/vm/drop_caches'],timeout=120)
        argv=make_argv(r,p,image,labels_for(r,epoch,source,phash,image),checks[r]['iface'],nccl)
        cid=remote(r,argv,timeout=90).strip();assert len(cid)==64
        record['ranks'].append(dict(rank=r,host=HOSTS[r],container_id=cid,argv=argv,labels=labels_for(r,epoch,source,phash,image)))
        save(root/'launch.json',record);print('STARTED',r,cid,flush=True)
    record['ranks'].sort(key=lambda x:x['rank']);save(root/'launch.json',record)
    # Guard is controller-owned and runs before expensive readiness.
    run([sys.executable,str(ROOT/'scripts/final_guard.py'),'start','--run-dir',str(root),'--image-id',image],timeout=120)
    deadline=time.monotonic()+3000
    while time.monotonic()<deadline:
        states=[]
        for row in record['ranks']:
            d=json.loads(remote(row['rank'],['docker','inspect',row['container_id']]))[0]
            save(root/f'rank{row["rank"]}-inspect.json',d)
            log=subprocess.run([*SSH,row['host'],shlex.join(['docker','logs','--tail','150',row['container_id']])],capture_output=True,text=True,timeout=20)
            (root/f'rank{row["rank"]}.log').write_text(log.stdout+log.stderr)
            assert d['Id']==row['container_id'] and d['Image']==image and d['State']['Running'],f'rank{row["rank"]} stopped'
        try:
            models=get('v1/models');info=get('get_server_info')
            if {x['id'] for x in models['data']}=={'/model'} and int(info['context_length'])==int(p['CONTEXT_LENGTH']):
                save(root/'server-info.json',info);save(root/'models.json',models);save(root/'READY.json',dict(epoch=epoch,image_id=image,time=time.time()));print('READY',str(root),flush=True);return
        except (OSError,ValueError,KeyError):pass
        time.sleep(15)
    raise RuntimeError('readiness deadline exceeded; inspect actual rank logs before action')


def stop(args):
    root=pathlib.Path(args.run_dir).resolve();d=json.loads((root/'launch.json').read_text())
    for _ in range(2):
        if not idle(get('get_load')):raise RuntimeError('unproven idle; refusing stop')
        time.sleep(1)
    # Stop only guards that this launch owns, before container teardown.
    run([sys.executable,str(ROOT/'scripts/final_guard.py'),'stop','--run-dir',str(root),'--image-id',d['image_id']],timeout=80)
    def one(row):
        r=row['rank'];cid=row['container_id'];cur=json.loads(remote(r,['docker','inspect',cid]))[0]
        assert cur['Id']==cid and cur['Image']==d['image_id']
        for k,v in row['labels'].items():assert cur['Config']['Labels'].get(k)==v
        remote(r,['docker','stop','--time','30',cid],timeout=45)
        after=json.loads(remote(r,['docker','inspect',cid]))[0];assert not after['State']['Running']
        save(root/f'rank{r}-stopped.json',after)
        logs=subprocess.run([*SSH,row['host'],shlex.join(['docker','logs','--timestamps',cid])],capture_output=True,text=True,timeout=30)
        (root/f'rank{r}-full.log').write_text(logs.stdout+logs.stderr)
        remote(r,['docker','rename',cid,'dsv41-final-archive-'+cid[:12]])
    with concurrent.futures.ThreadPoolExecutor(4) as ex:list(ex.map(one,d['ranks']))
    save(root/'STOPPED.json',dict(time=time.time(),epoch=d['epoch']))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['launch','stop']);ap.add_argument('--profile',default='prod');ap.add_argument('--run-dir',required=True);ap.add_argument('--image-id',default=os.environ.get('DSV41_FINAL_IMAGE_ID'));ap.add_argument('--dry-run',action='store_true');a=ap.parse_args()
    if a.mode=='launch' and not a.image_id:ap.error('--image-id required')
    (launch if a.mode=='launch' else stop)(a)
if __name__=='__main__':main()
