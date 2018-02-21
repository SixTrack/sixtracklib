import numpy as np

class CProp(object):
    def __init__(self,valuetype=None,offset=None,
                      default=0,const=False,
                      length=None):
        self.valuetype=valuetype
        self.offset=offset
        self.length=length
        self.default=default
        self.const=const
    def resolve_length(self,const):
        if isinstance(self.valuetype,CObject):
            size=self.valuetype.get_length(const)
        else:
            size=1
        if isinstance(self.length,str):
            length=eval(self.length,{},const)
        else:
            length=self.length
        if length is not None:
            length*=size
        return length
    def __get__(self,obj,type=None):
        name,offset,length=obj._attr[self.offset]
        shape=obj._shape.get(name)
        if length is None:
           val=obj._cbuffer.data[self.valuetype][offset]
        else:
           val=obj._cbuffer.data[self.valuetype][offset:offset+length]
           if shape is not None:
               val=val.reshape(*shape)
        return val
    def __set__(self,obj,val):
        name,offset,length=obj._attr[self.offset]
        shape=obj._shape.get(name)
        if not self.const:
           if length is None:
              obj._cbuffer.data[self.valuetype][offset]=val
           else:
              if shape is None:
                obj._cbuffer.data[self.valuetype][offset:offset+length]=val
              else:
                self.__get__(obj)[:]=val
        else:
           raise ValueError('property read-only')


class CBuffer(object):
    value_t = np.dtype({'names'  :['f64','i64','i32','u64','tag'],
                        'formats':['<f8','<i8','<i4','<u8','<u2'],
                        'offsets':[    0,    0,    0,    0,    6],
                        'itemsize':8})
    def __init__(self,initial_size):
        self.allocate_data(initial_size)
        self.next = 0 #index to first free data slot
    def allocate_data(self,size):
        self.size = size
        self.data = np.zeros(size, self.value_t)
        self.data_f64 = self.data['f64']
        self.data_i64 = self.data['i64']
        self.data_u64 = self.data['u64']
    def reserve_memory(self, size):
        minsize = self.next + size
        currsize = self.size
        if minsize > currsize:
            while minsize > currsize:
                currsize *= 2
            olddata = self.data
            self.allocate_data(currsize)
            self.data[:self.next] = olddata[:self.next]
    def add_values(self, data, datatype):
        size = len(data)
        dtype = self.value_t.fields[datatype][0]
        self.reserve_memory(size)
        self.data[datatype][self.next:self.next+size] = data
        self.next += size
    def add_float(self,data):
        self.add_values([data],'f64')
    def add_uinteger(self,data):
        self.add_values([data],'u64')
    def add_integer(self,data):
        self.add_values([data],'i64')
    def new_object(self):
        objid = self.next
        return objid


class CObject(object):
    def __init__(self, cbuffer = None, **nvargs):
        self._size = self._get_size(nvargs)
        if cbuffer is None:
            self._cbuffer=CBuffer(self._size)
        else:
            self._cbuffer=cbuffer
        self._offset=self._cbuffer.new_object()
        self._cbuffer.reserve_memory(self._size)
        self._cbuffer.next+=self._size
        self._fill_args(nvargs)
        self._shape={}
    def _get_props(self):
        props=[(prop.offset,name,prop) for name,prop
                                       in self.__class__.__dict__.items()
                                       if isinstance(prop,CProp)        ]
        props.sort()
        return props
    def _get_size(self,nvargs):
        props=self._get_props()
        const={}
        for offset,name,prop in props:
            if prop.const:
                const[name]=nvargs[name]
        size=0
        self._lengths={}
        self._names=[]
        for offset,name,prop in props:
            length=prop.resolve_length(const)
            if length is None:
                size+=1
            else:
                size+=length+1
            self._lengths[name]=length
            self._names.append(name)
        return size
    def _fill_args(self,nvargs):
        props=self._get_props()
        self._attr=[]
        lastarray=props[-1][0]+1
        for offset,name,prop in props:
            length=self._lengths[name]
            if isinstance(prop.valuetype,str):
              if length is None:
                 attr_offset=self._offset+offset
                 self._attr.append((name,attr_offset,None))
                 self._cbuffer.data[prop.valuetype][attr_offset]=  \
                                nvargs.get(name,prop.default)
              else:
                 attr_offset=self._offset+lastarray
                 self._attr.append((name,attr_offset,length))
                 self._cbuffer.data['u64'][self._offset+offset]=lastarray
                 lastarray+=length
                 self._cbuffer.data[prop.valuetype][attr_offset:attr_offset+length]= \
                                   nvargs.get(name,prop.default)
            elif isinstance(prop.valuetype,CObject):
                raise NotImplemented('Nested object not implemented')
    def pretty_print(self):
        props=self._get_props()
        out=['%s('%(self.__class__.__name__)]
        for offset,name,prop in props:
            val='%s'%getattr(self,name)
            out.append('  %-8s = %s,'%(name,val))
        out.append(')')
        return '\n'.join(out)
    def __repr__(self):
        return self.pretty_print()
