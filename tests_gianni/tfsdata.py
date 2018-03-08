"""Read and write data in the TFS format
Usage:
  data=load(fh):  Load data from a file object
  data=open(fn):  Load data from a file name
  dump(data,fh):  Dump data in file object
"""


from numpy import array

def _fromtfs(t,i):
  if ('e' in t):
    return float(i)
  elif ('s' in t):
    if i[0]=='"':
      return i[1:-1]
    else:
      return i
  elif ('d' in t):
    return int(i)



def _topythonname(n):
  n=n.lower()
  if n in ('param','param_names','col_names'):
    n+='__'
  return n

def _frompythonname(n):
  n=n.upper()
  if n in ('param__','param_names__','col_names__'):
    n=n[:-2]
  return n


def load(fh):
  """Load data encoded in the TFS format from a file object
  Usage:
    data=load(fh)
  """
  param={}
  datalines=0
  param_names=[]
  data={}
  col_names=[]
  plabels=[]
  for line in fh:
    line=line.strip()
    if line:
      lead=line[0]
      if (lead == '@'):  # descriptor lines
        f=line.split(None,3)
        try:
          param[_topythonname(f[1])]=_fromtfs(f[2],f[3])
        except:
          print("bad descriptor"," ".join(f))
        param_names.append(f[1])
      elif ( lead == '*'): # labels lines
        f=line.split()
        f.pop(0)
        col_names=f
        plabels=list(map(_topythonname,col_names))
        for l in plabels: data[l]=[]
      elif (lead == '$'):  # type lines
        f=line.split()
        f.pop(0) ; types=f
      elif (lead == '#'):  # comment lines
        pass
      else :   # data lines
        f=line.split()
        f=list(map(_fromtfs,types,f))
        datalines+=1
        for l in plabels:
          d=f.pop(0)
          data[l].append(d)
  len=datalines
  data.update(param=param,col_names=col_names,param_names=param_names)
  for l in plabels:
    data[l]=array(data[l])
  return data



def _totfstypes(i):
  if isinstance(i,float):
    return '%le'
  elif isinstance(i,int):
    return '%d'
  else:
    return '%s'

def dump(data,fh):
  """Dump data encoded in the TFS format in a file object
  Usage:
    dump(data,fh)
  """
  param=data['param']
  param_names=data.get('param_names',list(param.keys()))
  col_names=data['col_names']
  for k in param_names:
    v=param[_topythonname(k)]
    if isinstance(v,float):
      v='%19s' % v
      t= '%le'
    elif isinstance(v,int):
      v='%19s' % v
      t='%d'
    else:
      v=str(v)
      t='%%%02ds' % (len(v))
      v='"%s"' % v
    fh.write('@ %-16s %s %s\n' % (k,t,v))
  fh.write('* ')
  firstlabel=_topythonname(col_names[0])
  length=len(data[firstlabel])
  types=[]
  vec=[]
  lprint=[]
  for l in col_names:
    cur=data[_topythonname(l)]
    assert length==len(cur), \
       'len(%s)=%d differs from len(%s)=%d' % (l,len(cur),l[0],length)
    if isinstance(cur[0],str):
      cur=[ '"%s"' % i for i in cur]
    if isinstance(cur[0],str):
      cur=array(cur,dtype=str)
      lprint.append('%%-%ds' % (cur.dtype.itemsize+4))
      types.append(lprint[-1] % '%s')
    elif isinstance(cur[0],float):
      cur=array(cur,dtype='|S19')
      lprint.append('%%%ds' % (cur.dtype.itemsize))
      types.append(lprint[-1] % '%le')
    elif isinstance(cur[0],int):
      cur=array(cur,dtype=str)
      types.append(lprint[-1] % '%d')
      lprint.append('%%%ds' % (cur.dtype.itemsize))
    vec.append(cur)
    fh.write(lprint[-1] % l)
  fh.write('\n')
  fh.write('$ ')
  for t in types:
    fh.write(t)
  fh.write('\n')
  for i in range(length):
    fh.write(' ')
    for  cur,fmt in zip(vec,lprint):
      fh.write(fmt % cur[i])
    fh.write('\n')

def dump_csv(data,fh):
  """Dump data from TFS data into CSV
  Usage:
    dump_csv(data,fh)
  """
  def myrepr(v):
    return repr(v).replace("'",'"')
  param=data['param']
  param_names=data.get('param_names',list(param.keys()))
  col_names=data['col_names']
  for k in param_names:
    v=param[_topythonname(k)]
    fh.write('%s,%s\n' % (myrepr(k),myrepr(v)))
  fh.write(','.join(myrepr(l) for l in col_names)+'\n')
  firstlabel=_topythonname(col_names[0])
  length=len(data[firstlabel])
  for i in range(length):
    for l in col_names:
      val=data[_topythonname(l)][i]
      fh.write("%s, "%myrepr(val))
    fh.write('\n')



import gzip
def open(fn):
  """Load data encoded in the TFS format from a file name
  Usage:
    data=load(fn)
  """
  if fn.endswith('.gz'):
    fh=gzip.open(fn)
  else:
    fh=file(fn)
  t=load(fh)
  t['filename']=fn
  return t

def save(data,fn):
  """Load data encoded in the TFS format from a file name
  Usage:
    data=load(fn)
  """
  if fn.endswith('.gz'):
    fh=gzip.open(fn,'w')
  else:
    fh=file(fn,'w')
  dump(data,fh)


if __name__=='__main__':
  fh=file('test/tfsload.tfs')
  d=load(fh)
  fh.close()
  fh=file('test/tfswrite.tfs','w')
  dump(d,fh)
  fh.close()
