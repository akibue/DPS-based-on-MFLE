import concurrent.futures
import os

import numpy
import picos
from matplotlib import pyplot
import time

# local dimension of the canonical subspace
d = 2

#################################START
#Preparation for canonical subspace
# unnormalized MES
_MES = numpy.zeros((d**2,1))
for i in range(d):
	_MES[i*d+i] = 1.0
MES = picos.Constant("MES",_MES,(d**2,1))

# isometry spanning MES complement subspace
_VMESC = numpy.zeros((d**2,d-1),dtype=numpy.complex128)
for i in range(d):
	for j in range(1,d):
		_VMESC[i*d+i,j-1] = numpy.exp(2j*numpy.pi*i*j/d)
VMESC = picos.Constant("VMESC",_VMESC,(d**2,d-1))

# isometry spanning non MES subspace
_VNMES = numpy.zeros((d**2,d**2-d),dtype=numpy.complex128)
for i in range(d**2-d):
	_VNMES[i+1+i//d,i] = 1.0
VNMES = picos.Constant("VNMES",_VNMES,(d**2,d**2-d))

# isometry spanning complement of MES
VCMES = picos.Constant("VCMES",(VMESC & VNMES),(d**2,d**2-1))
#################################END

# isometry spanning the canonical subspace and MFLE (R_A R_B A B)
Vp = picos.Constant("Vp",(MES@MES & VCMES@VCMES),(d**4,d**4-2*(d**2-1)))
V = picos.Constant("V",(MES@MES & VCMES@picos.I(d**2)),(d**4,d**4-(d**2-1)))

#################################START
#Preparation for symmetric extension
#SWAP A B -> B A
# local dimensions
dA = 2
dB = 2

_CBA = numpy.zeros((dA,dA),dtype=numpy.complex128)
for i in range(dA):
	_CBA[i,i]=1.0

_CBB = numpy.zeros((dB,dB),dtype=numpy.complex128)
for i in range(dB):
	_CBB[i,i]=1.0

_SWAP = numpy.zeros((dA*dB,dA*dB),dtype=numpy.complex128)
SWAP = picos.Constant(_SWAP)
for i in range(dA):
	for j in range(dB):
		SWAP = SWAP + (picos.Constant("CB",_CBB[j],(dB,1))@picos.Constant("CA",_CBA[i],(dA,1)))*(picos.Constant("CA",_CBA[i],(1,dA))@picos.Constant("CB",_CBB[j],(1,dB)))
		
# dimension of local system (R_B B) forming the symmetric subspace
dl = d**2
SymD = int(dl*(dl+1)/2)

# operator spanning symmetric subspace
VSYM = picos.Constant("VSYM",[[1.0,0,0,0,0,0,0,0,0,0],[0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0],[0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0],[0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0],[0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0],[0,0,0,0,1.0,0,0,0,0,0],[0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0],[0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0],[0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0],[0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0],[0,0,0,0,0,0,0,1.0,0,0],[0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0],[0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0],[0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0],[0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0],[0,0,0,0,0,0,0,0,0,1.0]],(dl**2,SymD))

#################################END

#################################
#INITIALIZATION
#################################START
tstart = 0.05*numpy.pi
tend = 0.25*numpy.pi
div = 99

#epsilon-net
Enet = 400
Enetc = 340

# size of epsilon-net for remnant CP map
#Enet=Enetc=400 fsb-3, ipm-7 upper bound < lower bound
#Enet=Enetc=390 fsb3*10^(-4), ipm3*10^(-8) upper bound < lower bound  3/4 instances
#Enet=Enetc=375 fsb2*10^(-4), ipm1*10^(-8) 
#Enet=Enetc=300 fsb-4, ipm-8: 2% error 
#Enet=Enetc=350 fsb-4, ipm-8: 1% error jaggie and upper bound < lower bound
#Enet=Enetc=400 fsb-5, ipm-8: % error too many error
#Enet=Enetc=375 fsb-5, ipm-8: 0% error upper bound < lower bound 
#Enet=Enetc=380 fsb-5, ipm-8: % error upper bound < lower bound 
#################################END

#################################
#MFLE upper bound
#################################
def fMFLE(t1):
	# resource entangled state (|τ>=I@T1*MES) R_AR_B
	tau = picos.Constant("τ",[numpy.cos(t1),0,0,numpy.sin(t1)],(4,1))
	T1 = picos.Constant("T1",[numpy.cos(t1),0,0,numpy.sin(t1)],(2,2))

	# target state (|φ>=I@(T2)*MES) AB
	p1 = (numpy.sqrt(3)+1)/(2*numpy.sqrt(2))
	p2 = (numpy.sqrt(3)-1)/(2*numpy.sqrt(2))
	T2 = picos.Constant("T2",[p1,0,0,-p2],(2,2))
	
	# isometry spanning the canonical subspace (R_A R_B A B)
	Wp = picos.Constant("Wp",(picos.I(d)@(numpy.linalg.inv(T1.H))@picos.I(d)@T2)*Vp,((d**4),(Vp.shape[1])))

	# parameter matrix of S
	MatS = picos.HermitianVariable("MatS",Vp.shape[1])

	# SDP for PPT+MFLE relaxation of the success probability
	PPTprob = picos.Problem()
	PPTprob.set_objective("max",MatS[0,0])
	PPTprob.add_constraint(MatS >> 0)
	PPTprob.add_constraint((Wp*MatS*Wp.H).partial_transpose((0,2)) >> 0)
	PPTprob.add_constraint(picos.I(d**4) - Wp*MatS*Wp.H >> 0)
	PPTprob.add_constraint(picos.I(d**4) - (Wp*MatS*Wp.H).partial_transpose((0,2)) >> 0)

	PPTprob.solve(solver = "qics")

	print("(θ,MFLE)=", t1, PPTprob.value*d**2)
	return PPTprob.value*d**2

#################################
#PPT upper bound
#################################
def fPPT(t1):
	# resource entangled state (|τ>=I@T1*MES) R_AR_B
	tau = picos.Constant("τ",[numpy.cos(t1),0,0,numpy.sin(t1)],(4,1))
	T1 = picos.Constant("T1",[numpy.cos(t1),0,0,numpy.sin(t1)],(2,2))

	# target state (|φ>=I@(T2)*MES) AB
	p1 = (numpy.sqrt(3)+1)/(2*numpy.sqrt(2))
	p2 = (numpy.sqrt(3)-1)/(2*numpy.sqrt(2))
	T2 = picos.Constant("T2",[p1,0,0,-p2],(2,2))

	# isometry spanning the canonical subspace (R_A R_B A B)
	W = picos.Constant("W",(picos.I(d)@(numpy.linalg.inv(T1.H))@picos.I(d)@T2)*V,((d**4),(V.shape[1])))

	# parameter matrix of S
	MatS = picos.HermitianVariable("MatS",V.shape[1])

	# SDP for PPT relaxation of the success probability
	PPTprob = picos.Problem()
	PPTprob.set_objective("max",MatS[0,0])
	PPTprob.add_constraint(MatS >> 0)
	PPTprob.add_constraint((W*MatS*W.H).partial_transpose((0,2)) >> 0)
	PPTprob.add_constraint(picos.I(d**4) - W*MatS*W.H >> 0)
	PPTprob.add_constraint(picos.I(d**4) - (W*MatS*W.H).partial_transpose((0,2)) >> 0)

	PPTprob.solve(solver = "qics")

	print("(θ,PPT)=", t1, PPTprob.value*d**2)
	return PPTprob.value*d**2

#################################
#DPS 2nd Lv upper bound
#################################
def fDPS2(t1):
	# resource entangled state (|τ>=I@T1*MES) R_AR_B
	tau = picos.Constant("τ",[numpy.cos(t1),0,0,numpy.sin(t1)],(4,1))
	T1 = picos.Constant("T1",[numpy.cos(t1),0,0,numpy.sin(t1)],(2,2))

	# target state (|φ>=I@(T2)*MES) AB
	p1 = (numpy.sqrt(3)+1)/(2*numpy.sqrt(2))
	p2 = (numpy.sqrt(3)-1)/(2*numpy.sqrt(2))
	T2 = picos.Constant("T2",[p1,0,0,-p2],(2,2))

	# local linear transformation  (A RA B RB)
	W = picos.Constant("W",picos.I(4)@T2@(numpy.linalg.inv(T1.H)),(d**4,d**4))

	# parameter matrix of S (VDPS*S has range in A'  RA'  A  RA  B  RB)
	MatS = picos.HermitianVariable("MatS",28)

	ExtMat = picos.HermitianVariable("ExtMat",SymD*d**2)

	# isometry for PPT operation (A'  RA'  A  RA  B  RB)
	VDPS = picos.Constant(
	[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,-(1.0/numpy.sqrt(2.0)),0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,-(1.0/numpy.sqrt(2.0)),0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-(1.0/numpy.sqrt(2.0)),0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0,0],[0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,-1.0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0],[0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,1.0,0,0,0,0,0,0,0,-(1.0/numpy.sqrt(2.0)),0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,-(1.0/numpy.sqrt(2.0)),0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0],[1.0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,-(1.0/numpy.sqrt(2.0)),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,-(1.0/numpy.sqrt(2.0)),0,0,0,-1.0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,-(1.0/numpy.sqrt(2.0)),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,-1.0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,-(1.0/numpy.sqrt(2.0)),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,1.0,0,0,0,0,0,0,0,-(1.0/numpy.sqrt(2.0)),0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,-(1.0/numpy.sqrt(2.0)),0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,-(1.0/numpy.sqrt(2.0)),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,-(1.0/numpy.sqrt(2.0)),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,-1.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1.0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,-(1.0/numpy.sqrt(2.0)),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,-(1.0/numpy.sqrt(2.0)),0,0,0,-1.0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,-(1.0/numpy.sqrt(2.0)),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,-1.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,1.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,1.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,1.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

	# SDP for PPT relaxation of the success probability
	PPTprob = picos.Problem()
	PPTprob.set_objective("max",0.5*(T2*T2.H).tr*(2.0*MatS[0,0]+numpy.sqrt(2.0)*MatS[0,5]+MatS[4,4]+numpy.sqrt(2.0)*MatS[4,8]+numpy.sqrt(2.0)*MatS[5,0]+MatS[5,5]+numpy.sqrt(2.0)*MatS[8,4]+2.0*MatS[8,8]+MatS[10,10]+MatS[10,15]+MatS[14,14]-MatS[14,21]+MatS[15,10]+MatS[15,15]-MatS[21,14]+MatS[21,21]))
	PPTprob.add_constraint(MatS >> 0)
	PPTprob.add_constraint((VDPS*MatS*VDPS.H).partial_transpose((0),(4,4,4)) >> 0)
	PPTprob.add_constraint((VDPS*MatS*VDPS.H).partial_transpose((2),(4,4,4)) >> 0)

	PPTprob.add_constraint(ExtMat >> 0)
	PPTprob.add_constraint(((VSYM@picos.I(d**2))*ExtMat*(VSYM@picos.I(d**2)).H).partial_transpose((0),(4,4,4)) >> 0)
	PPTprob.add_constraint(((VSYM@picos.I(d**2))*ExtMat*(VSYM@picos.I(d**2)).H).partial_transpose((2),(4,4,4)) >> 0)
	PPTprob.add_constraint(picos.I(d**4) >> W*(VDPS*MatS*VDPS.H).partial_trace((0),(4,4,4))*W.H + ((VSYM@picos.I(d**2))*ExtMat*(VSYM@picos.I(d**2)).H).partial_trace((0),(4,4,4)))
	PPTprob.add_constraint(picos.I(d**4) << W*(VDPS*MatS*VDPS.H).partial_trace((0),(4,4,4))*W.H + ((VSYM@picos.I(d**2))*ExtMat*(VSYM@picos.I(d**2)).H).partial_trace((0),(4,4,4)))
	#PPTprob.add_constraint(picos.I(d**4) - W*(VDPS*MatS*VDPS.H).partial_trace((2),(4,4,4))*W.H >> 0)
	#PPTprob.add_constraint(picos.I(d**4) - (W*(VDPS*MatS*VDPS.H).partial_trace((2),(4,4,4))*W.H).partial_transpose((0),(4,4)) >> 0)

	PPTprob.solve(solver = "qics")

	print("(θ,DPS2)=", t1, PPTprob.value)
	return PPTprob.value

#################################
#SEP lower bound
#################################
def fSEP(t1):
	# resource entangled state (|τ>=I@T1*MES) R_AR_B
	tau = picos.Constant("τ",[numpy.cos(t1),0,0,numpy.sin(t1)],(4,1))
	T1 = picos.Constant("T1",[numpy.cos(t1),0,0,numpy.sin(t1)],(2,2))

	# target state (|φ>=I@(T2)*MES) AB
	p1 = (numpy.sqrt(3)+1)/(2*numpy.sqrt(2))
	p2 = (numpy.sqrt(3)-1)/(2*numpy.sqrt(2))
	T2 = picos.Constant("T2",[p1,0,0,-p2],(2,2))

	#e-net
	matA = [picos.Constant("A[{}]".format(neti),numpy.random.randn(d**2,2).view(numpy.complex128),(d,d)) for neti in range(Enet)]	
	matBt = [picos.Constant("B[{}]".format(neti),numpy.linalg.inv(matA[neti]),(d,d)) for neti in range(Enet)]	
	A = [picos.Constant("A[{}]".format(neti),numpy.random.randn(d**2,2).view(numpy.complex128),(d**2,1)) for neti in range(Enetc)]	
	RemAstate = [picos.Constant("RemA[{}]".format(neti),A[neti]*A[neti].H/(A[neti]*A[neti].H).tr,(d**2,d**2)) for neti in range(Enetc)]

	# vector in S∩V (R_A A R_B B)
	SVVec = [picos.Constant("Vec[{}]".format(neti),((picos.I(d)@matA[neti])*MES)@(((numpy.linalg.inv(T1.H))@(T2*matBt[neti].reshuffled("ji")))*MES),(d**4,1)) for neti in range(Enet)]

	# variables for epsilon-net
	posvar = picos.RealVariable("posvar",Enet)

	Dist = picos.RealVariable("δ")

	# SDP for computing the Schatten L1 norm between Butone and discretized SEP
	RemB = [picos.HermitianVariable("RemB[{}]".format(neti),d**2) for neti in range(Enetc)]
	trho = picos.HermitianVariable("tρ",d**4)
	
	# SDP for discretized SEP 
	SEPprob = picos.Problem()
	SEPprob.set_objective("max",picos.sum(posvar[neti] for neti in range(Enet)) - Dist)
	SEPprob.add_constraint(posvar >=0)
	SEPprob.add_constraint(Dist >= 0)
	SEPprob.add_constraint(Dist >= 2.0*trho.tr + picos.sum(RemB[neti].tr for neti in range(Enetc)) - d**4 +  picos.sum(posvar[neti]*(SVVec[neti]*SVVec[neti].H).tr for neti in range(Enet)))

	SEPprob.add_constraint(trho >> 0)
	SEPprob.add_constraint(trho >> picos.I(d**4) - picos.sum(posvar[neti]*SVVec[neti]*SVVec[neti].H for neti in range(Enet)) - picos.sum(RemAstate[neti]@RemB[neti] for neti in range(Enetc)))
	SEPprob.add_list_of_constraints([RemB[neti] >> 0 for neti in range(Enetc)])
	
	SEPprob.options["rel_prim_fsb_tol"] = 1*10**(-5)
	SEPprob.options["rel_dual_fsb_tol"] = 1*10**(-5)
	SEPprob.options["rel_ipm_opt_tol"] = 1*10**(-8)

	try:
		SEPprob.solve(solver = "qics")
	except Exception as e:
		print(f"Error: {e}")
		return -1
			
	SEPBObound = SEPprob.value + Dist.value
	print("(θ,SEP but one)=", t1, SEPBObound)
		
	print("(θ,Distance between RemCP and SEP):", t1, Dist.value)	
	return SEPBObound/(1.0 + Dist.value)

	
#################################
#MAIN
if __name__ == "__main__":
	t1 = [tstart + (tend-tstart)*step/div for step in range(div+1)]
	MFLEstartTime = time.time()
	with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
		futures = [executor.submit(fMFLE, t1[step]) for step in range(div+1)]
		MFLEbound = [f.result() for f in futures]
	MFLEendTime = time.time()

	PPTstartTime = time.time()
	with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
		futures = [executor.submit(fPPT, t1[step]) for step in range(div+1)]
		PPTbound = [f.result() for f in futures]
	PPTendTime = time.time()

	DPS2startTime = time.time()
	with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
		futures = [executor.submit(fDPS2, t1[step]) for step in range(div+1)]
		DPS2bound = [f.result() for f in futures]
	DPS2endTime = time.time()

	SEPstartTime = time.time()
	with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
		futures = [executor.submit(fSEP, t1[step]) for step in range(div+1)]
		SEPbound = [f.result() for f in futures]
	SEPendTime = time.time()


	print("PPT + MFLE time:",(MFLEendTime-MFLEstartTime)/(div+1))
	print("PPT time:",(PPTendTime-PPTstartTime)/(div+1))
	print("DPS2 time:",(DPS2endTime-DPS2startTime)/(div+1))
	print("SEP time:",(SEPendTime-SEPstartTime)/(div+1))

	SEPboundc = []
	tc = []
	for step in range(div+1):
		if SEPbound[step] >= 0:
			tc.append(t1[step])
			SEPboundc.append(SEPbound[step])

	with open('theta.txt', mode='w') as f:
    		f.write('\n'.join([str(t1[step]) for step in range(div+1)]))

	with open('MFLE.txt', mode='w') as f:
    		f.write('\n'.join([str(MFLEbound[step]) for step in range(div+1)]))

	with open('PPT.txt', mode='w') as f:
    		f.write('\n'.join([str(PPTbound[step]) for step in range(div+1)]))

	with open('DPS2nd.txt', mode='w') as f:
    		f.write('\n'.join([str(DPS2bound[step]) for step in range(div+1)]))

	with open('lowerbound.txt', mode='w') as f:
    		f.write('\n'.join([str(SEPbound[step]) for step in range(div+1)]))

	
	#pyplot.title("")
	pyplot.xlabel("θ")
	pyplot.ylabel("q(φ,τ)")
	pyplot.plot(t1,PPTbound, label='PPT (DPS 1st Lv.)')
	pyplot.plot(t1,DPS2bound, label='DPS 2nd Lv.')
	pyplot.plot(tc,SEPboundc, label='lower bound')
	pyplot.plot(t1,MFLEbound, label='PPT + MFLE')
	pyplot.legend()
	pyplot.show()
