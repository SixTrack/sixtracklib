#!/usr/bin/env python


try:
	from Numeric import *
	from LinearAlgebra import *
except:
	from numpy import *
	from numpy import dot as matrixmultiply
#from numpy.linalg import inv as generalized_inverse
        from numpy.linalg import inv as inverse
        from numpy.linalg import det as determinant


from string import split
from string import replace
import sys
import gzip


#########################
class twiss:
#########################
    "Twiss parameters from madx output (with free choice of select items)"

    def forknames(self, dictionary):
        for n in dictionary:
            if n in self.NAME:
                for m in  dictionary[n]:
                    self.indx[m]=self.indx[n]
                    self.indx[m.upper()]=self.indx[n]
            else:
                print n, "from dictionary not in NAME; ",n," skiped "
                
    def __init__(self, filename, dictionary={}): 
        self.indx={}
        self.keys=[]
        alllabels=[]
        if '.gz' in filename:
            f=gzip.open(filename, 'rb')
        else:
            f=open(filename, 'r')
            
        for line in f:
#            if ("@ " in line and "%le" in line) :      # FIX to take DPP %s
            if ("@ " not in line and "@" in line): 
              line = replace(line, "@" , "@ ")
            if ("@ " in line and "%" in line and "s" not in split(line)[2]) :
                label=split(line)[1]
		try:
                	exec "self."+label+"= "+str(float(split(replace(line, '"', ''))[3]))
		except:
			print "Problem parsing:", line
			print "Going to be parsed as string"
                        try:
                            exec "self."+label+"= \""+replace(split(line)[3], '"', '')+"\""
                        except:
                            print "Problem persits, let's ignore it!"
            elif ("@ " in line and "s"  in split(line)[2]):
                label=split(line)[1].replace(":","")
                exec "self."+label+"= \""+split(replace(line, '"', ''))[3]+"\""


	    if ("* " in line or "*\t" in line) :
                alllabels=split(line)
                print "alllabels",len(alllabels)
                for j in range(1,len(alllabels)):
                    exec "self."+alllabels[j]+"= []"
                    self.keys.append(alllabels[j])
                            
            if ("$ " in line or "$\t" in line) :
                alltypes=split(line)


                

            if ("@" not in line and "*" not in line and "$" not in line) :
                values=split(line)
                for j in range(0,len(values)):
                    if ("%hd" in alltypes[j+1]):                      
                      exec "self."+alllabels[j+1]+".append("+str(int(values[j]))+")"
                    
                    if ("%le" in alltypes[j+1]):                      
                      exec "self."+alllabels[j+1]+".append("+str(float(values[j]))+")"
                    if ("s" in alltypes[j+1]):
                      try:
                        exec "self."+alllabels[j+1]+".append("+values[j]+")"
                      except:
                        exec "self."+alllabels[j+1]+".append(\""+values[j]+"\")" #To allow with or without ""
                      if "NAME"==alllabels[j+1]:
                      	self.indx[replace(values[j], '"', '')]=len(self.NAME)-1
                      	self.indx[replace(values[j], '"', '').upper()]=len(self.NAME)-1
                        self.indx[replace(values[j], '"', '').lower()]=len(self.NAME)-1

        f.close()
        try:
            alllabels
            alltypes
        except:
            print "From Metaclass: Bad format or empy file ", filename
            print "Leaving Metaclass"
            exit()

        
        for j in range(1,len(alllabels)):
            if (("%le" in alltypes[j]) | ("%hd" in alltypes[j])  ):  
                exec "self."+alllabels[j]+"= array(self."+alllabels[j]+")"           

        if len(dictionary) > 0:
            self.forknames(dictionary)

    def chrombeat(self):
      self.dbx=[]
      self.dby=[]
      for i in range(0,len(self.S)): 
        ax=self.WX[i]*cos(self.PHIX[i]*2*pi)
        ay=self.WY[i]*cos(self.PHIY[i]*2*pi)
        self.dbx.append(ax)
        self.dby.append(ay)
        
    def fterms(self):
        self.f3000= []
        self.f2100= []
        self.f1020= []
        self.f1002= []
        self.f20001= []
        self.f1011= []
        self.f4000=[]
        self.f2000=[]
        for i in range(0,len(self.S)):
            phix = self.MUX-self.MUX[i]
            phiy = self.MUY-self.MUY[i]
            for j in range(0,i):
                phix[j] += self.Q1
                phiy[j] += self.Q2
            dumm=-sum(self.K2L*self.BETX**1.5*e**(3*complex(0,1)*2*pi*phix))/24.
            self.f3000.append(dumm)
            dumm=-sum(self.K2L*self.BETX**1.5*e**(complex(0,1)*2*pi*phix))/8.
            self.f2100.append(dumm)
            dumm=sum(self.K2L*self.BETX**0.5*self.BETY*e**(complex(0,1)*2*pi*(phix+2*phiy)))/8.
            self.f1020.append(dumm)
            dumm=sum(self.K2L*self.BETX**0.5*self.BETY*e**(complex(0,1)*2*pi*(phix-2*phiy)))/8.
            self.f1002.append(dumm)
            dumm=sum((self.K1L-2*self.K2L*self.DX)*self.BETX*e**(2*complex(0,1)*2*pi*phix))/8.
            self.f20001.append(dumm)
            dumm=sum(self.K2L*self.BETX**0.5*self.BETY*e**(complex(0,1)*2*pi*(phix)))/4.
            self.f1011.append(dumm)
            dumm=-sum(self.K3L*self.BETX**2*e**(4*complex(0,1)*2*pi*(phix)))/384.
            self.f4000.append(dumm)
            dumm=-sum(self.K1L*self.BETX**1*e**(2*complex(0,1)*2*pi*phix))/32.
            self.f2000.append(dumm)
            
    def chiterms(self, ListOfBPMS=[]):
        factMADtoSix=0.0005
        self.chi3000=[]
        self.chi4000=[]
        self.chi2000=[]
        if len(ListOfBPMS)==0:
            print "Assuming that BPM elements are named as BP and H"
            for el in self.NAME:
                if "BP" in el and "H" in el:
                    ListOfBPMS.append(el)
                    
        print "Found ", len(ListOfBPMS), "BPMs for chiterms computation"
        if len(ListOfBPMS)<3:
            print "Error, not enough H BPMs in ListOfBPMs"
            sys.exit()
        
        self.chi=[]
        self.chiBPMs=[]
        self.chiS=[]
        for i in range(len(ListOfBPMS)-2):
            name=ListOfBPMS[i]
            name1=ListOfBPMS[i+1]
            name2=ListOfBPMS[i+2]
            self.chiBPMs.append([name,name1,name2])
            indx=self.indx[name]
            indx1=self.indx[name1]
            indx2=self.indx[name2]
            bphmii=self.MUX[indx]
            bphmii1=self.MUX[indx1]
            bphmii2=self.MUX[indx2]
            bphs=self.S[indx]
            bphs1=self.S[indx1]
            bphs2=self.S[indx2]
            self.chiS.append([bphs,bphs1,bphs2])
            d1= (bphmii1- bphmii)*2*pi-pi/2;
            d2= (bphmii2- bphmii1)*2*pi-pi/2;
            f1= sqrt(1+(sin(d1)/cos(d1))**2);
            f2= sqrt(1+(sin(d2)/cos(d2))**2);
            quadr=0
            quadi=0
            sexr=0
            sexi=0
            octr=0
            octi=0
            for j in range(len(self.NAME)):
                k1l=self.K1L[j]
                k2l=self.K2L[j]
                k3l=self.K3L[j]
                bx=self.BETX[j]
                m=self.MUX[j]
                if self.S[j] > bphs and self.S[j] < bphs1 and k2l**2 > 0:

                    quadr+=cos(-1*(m-bphmii)*2*pi)*sin((m-bphmii)*2*pi)*k1l*bx**1*f1;
                    quadi+=sin(-1*(m-bphmii)*2*pi)*sin((m-bphmii)*2*pi)*k1l*bx**1*f1;
                    
                    sexr += cos(-2*(m-bphmii)*2*pi)*sin((m-bphmii)*2*pi)*k2l*bx**1.5*f1;
                    sexi += sin(-2*(m-bphmii)*2*pi)*sin((m-bphmii)*2*pi)*k2l*bx**1.5*f1;

                    octr += cos(-3*(m-bphmii)*2*pi)*sin((m-bphmii)*2*pi)*k3l*bx**2*f1;
                    octi += sin(-3*(m-bphmii)*2*pi)*sin((m-bphmii)*2*pi)*k3l*bx**2*f1;
                    
                if self.S[j] > bphs1 and self.S[j] < bphs2 and k2l**2 > 0:

                    quadr+=cos(-1*(m-bphmii)*2*pi)*sin((m-bphmii)*2*pi-d1-d2)*k1l*bx**1*f2;
                    quadi+=sin(-1*(m-bphmii)*2*pi)*sin((m-bphmii)*2*pi-d1-d2)*k1l*bx**1*f2;
                    
                    sexr += cos(-2*(m-bphmii)*2*pi)*sin((m-bphmii)*2*pi-d1-d2)*k2l*bx**1.5*f2;
                    sexi += sin(-2*(m-bphmii)*2*pi)*sin((m-bphmii)*2*pi-d1-d2)*k2l*bx**1.5*f2;

                    octr += cos(-3*(m-bphmii)*2*pi)*sin((m-bphmii)*2*pi-d1-d2)*k3l*bx**2*f2;
                    octi += sin(-3*(m-bphmii)*2*pi)*sin((m-bphmii)*2*pi-d1-d2)*k3l*bx**2*f2;
                    
                if self.S[j] > bphs2:
                    break
            self.chi.append(complex(sexr,sexi)/4*factMADtoSix)
            self.chi4000.append(complex(octr,octi)/4*factMADtoSix)
            self.chi2000.append(complex(quadr,quadi)/4*factMADtoSix)                
        


    def Cmatrix(self):
        self.C = []
        self.gamma = []
        self.f1001 = []
        self.f1010 = []
        
        J = reshape(array([0,1,-1,0]),(2,2))
        for j in range(0,len(self.S)):
            R = array([[self.R11[j],self.R12[j]],[self.R21[j],self.R22[j]]])
            #print R
            C = matrixmultiply(-J,matrixmultiply(transpose(R),J))
            C = (1/sqrt(1+determinant(R)))*C

            g11 = 1/sqrt(self.BETX[j])
            g12 = 0
            g21 = self.ALFX[j]/sqrt(self.BETX[j])
            g22 = sqrt(self.BETX[j])
            Ga = reshape(array([g11,g12,g21,g22]),(2,2))

            g11 = 1/sqrt(self.BETY[j])
            g12 = 0
            g21 = self.ALFY[j]/sqrt(self.BETY[j])
            g22 = sqrt(self.BETY[j])
            Gb = reshape(array([g11,g12,g21,g22]),(2,2))
	    C = matrixmultiply(Ga, matrixmultiply(C, inverse(Gb)))
            gamma=1-determinant(C)
            self.gamma.append(gamma)
            C = ravel(C)
            self.C.append(C)
            self.f1001.append(((C[0]+C[3])*1j + (C[1]-C[2]))/4/gamma)
            self.f1010.append(((C[0]-C[3])*1j +(-C[1]-C[2]))/4/gamma)

        self.F1001R=array(self.f1001).real
        self.F1001I=array(self.f1001).imag
        self.F1010R=array(self.f1010).real
        self.F1010I=array(self.f1010).imag
        self.F1001W=sqrt(self.F1001R**2+self.F1001I**2)
        self.F1010W=sqrt(self.F1010R**2+self.F1010I**2)
        
    def beatMatrix(self):
        self.RM = []
        for j in range(0,len(self.S)):
            self.RM.append(-self.BETX*cos(2*pi*(self.Q1-2*abs(self.MUX[j]-self.MUX)))/sin(2*pi*self.Q1))
        for j in range(0,len(self.S)):
            self.RM.append(-self.BETY*cos(2*pi*(self.Q2-2*abs(self.MUY[j]-self.MUY)))/sin(2*pi*self.Q2))
        self.RM=array(self.RM)



# Read the twiss class from the twiss file
#x=twiss('twiss')
# use it as:
#print x.Q1, x.Q2, x.BETX[0]


# run beaMatrix for example:
#x.beatMatrix()
#print x.RM[0]


# BETA-BEAT CORRECTION
# first compute the response matrix by:
# x.beatMatrix()
# Define targetbeat as an array containing the desired changed in Dbeta/beta (x,y)
# targetbeat=Dbeta/beta
# dkl gives the required integrated strengths by: 
# dkl=matrixmultiply(generalized_inverse(x.RM,0.003),targetbeat)


#Want to explore the singular values?:
#svd=singular_value_decomposition(x.RM)


# Computing SEXTUPOLAR RESONANCE TERMS:
# x.fterms()
# The fterms are arrays evaluated at all the elements:
# print x.f3000 , x.f2100  , x.f1020, x.f1002


# COUPLING
# Compute the Cmatrix, gamma, f1001 and f1010 from the Twiss file at all elements
# x.Cmatrix()
# print x.C[0]   (four components of C at the first elements)
# print x.f1001
# ...


