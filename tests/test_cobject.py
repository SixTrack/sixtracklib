from sixtracklib import  CProp, CObject, CBuffer


class A(CObject):
    a=CProp('f64',0,const=True)
    b=CProp('f64',1)
    c=CProp('f64',2,length='3*a+2')

class B(CObject):
    a=CProp('f64',0)
    b=CProp('f64',1)
    c=CProp('f64',2)


class C(CObject):
    a=CProp('f64',0)
    b=CProp(B,1)


def test_length():
    assert A._get_size(a=0)==5
    assert A._get_size(a=2)==11
    assert C._get_size()==4


def test_build():
    a=A(a=0)
    b=B()
    c=C(a=3,b={'a':1,'b':2,'c':3})
    assert a.a==0
    assert b.b==0
    assert c.a==3
    assert c.b.a==1
    assert c.b.c==3

