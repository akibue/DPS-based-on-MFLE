import concurrent.futures

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
tstart = 0.16*numpy.pi
tend = 0.25*numpy.pi
div = 99

#epsilon-net 150GB MEM/3000 2%
#Enet=2000: fsb-6 ipm-8 2% error 60GBMEM
#Enet=3000: fsb-6 ipm-8 2% error 140GBMEM
#Enet=1000, Enetc=400, fsb-4, ipm-8: 0 instances
#Enet=1000, Enetc=350, fsb2*10**-4, ipm2*10**-8: 0 instances
#Enet=1000, Enetc=350, fsb3*10**-4, ipm3*10**-8: 0 instances
#Enet=1000, Enetc=350, fsb4*10**-4, ipm4*10**-8: 0 instances
#Enet=1000, Enetc=350, fsb5*10**-4, ipm5*10**-8: 0 instances
#Enet=400, Enetc=350, fsb2*10**-4, ipm1*10**-8: 5% error 1/8 instances
#Enet=500, Enetc=300, fsb3*10**-4, ipm1.5*10**-8: 8% error  3/8 instances
#Enet=1000, Enetc=300, fsb3*10**-4, ipm1.5*10**-8: % error   instances
#Enet=500, Enetc=350, fsb2*10**-4, ipm1*10**-8: 5% error 5/6 instances

#Enet=400, Enetc=350, fsb10**-5, ipm10**-8: 5% error  1/3 instances 8min
#Enet=400, Enetc=350, fsb 2*10**-5, ipm10**-8: 5% error  1/2 instances 8min
#Enet=500, Enetc=350, fsb10**-4, ipm10**-8: % error  1/8 instances 8min lower bound > upper bound
#Enet=750, Enetc=325, fsb 2*10**-5, ipm10**-8: 3% error  3/8 instances 8 min
#Enet=1000, Enetc=325, fsb 3*10**-5, ipm10**-8: % error  0 instances  9min
#Enet=800, Enetc=330, fsb 3*10**-5, ipm10**-8: % error  1/6 instances  min
#Enet=800, Enetc=350, fsb 10**-5, ipm10**-8: % error  0 instances  9 min lower bound > upper bound
#Enet=800, Enetc=300, fsb 10**-5, ipm10**-8: % error  7/16 instances  9 min
#Enet=900, Enetc=310, fsb 2*10**-5, ipm10**-8: 2% error  6/16 instances 9 min
#Enet=1000, Enetc=310, fsb 3*10**-5, ipm10**-8: 2% error  7/8 instances 16 min 15GB MEM
#Enet=1000, Enetc=320, fsb 3*10**-5, ipm10**-8: 1% error  3/4 instances 16 min GB MEM
#Enet=1200, Enetc=320, fsb 3*10**-5, ipm10**-8: 0.5% error  1/1 instances  min 22 GB MEM lower bound < upper bound
#Enet=1500, Enetc=320, fsb 3*10**-5, ipm10**-8: % error   instances  min  GB MEM
#Enet=1000, Enetc=315, fsb 3*10**-5, ipm10**-8: % error  15 GB MEM Error if 8 parallel 1000 sec
#Enet=1150, Enetc=315, fsb 3*10**-5, ipm10**-8: % error  20 GB MEM sec lower < upper bound
#Enet=1100, Enetc=315, fsb 3*10**-5, ipm10**-8: % error  20GB MEM  sec lower < upper bound 
#Enet=1050, Enetc=315, fsb 3*10**-5, ipm10**-8: % error  20GB MEM  sec 
#Enet=1070, Enetc=315, fsb 3*10**-5, ipm10**-8: % error  20GB MEM  sec lower < upper

Enet = 1050

Enetc = 315

#################################END

#################################
#MFLE upper bound
#################################
def fMFLE(t1):
	# resource entangled state (|τ>=I@T1*MES) R_AR_B
	tau = picos.Constant("τ",[numpy.cos(t1),0,0,numpy.sin(t1)],(4,1))
	T1 = picos.Constant("T1",[numpy.cos(t1),0,0,numpy.sin(t1)],(2,2))

	# PVM basis (|φi>=I@(T2)*MES)
	p1 = (numpy.sqrt(3)+1)/(2*numpy.sqrt(2))
	p2 = (numpy.sqrt(3)-1)/(2*numpy.sqrt(2))
	zeta = numpy.exp(1j*numpy.pi/3)
	T2 = picos.Constant("T2",[p1,0,0,-p2],(2,2))
	T3 = picos.Constant("T3",[p2,1,1,p1]/numpy.sqrt(3),(2,2))
	T4 = picos.Constant("T4",[p2,-zeta,zeta**2,p1]/numpy.sqrt(3),(2,2))
	T5 = picos.Constant("T5",[p2,zeta**2,-zeta,p1]/numpy.sqrt(3),(2,2))
	
	# isometry spanning the canonical subspace (R_A R_B A B)
	Wp1 = picos.Constant("Wp1",(picos.I(d)@(numpy.linalg.inv(T1.H))@picos.I(d)@T2)*Vp,((d**4),(Vp.shape[1])))
	Wp2 = picos.Constant("Wp2",(picos.I(d)@(numpy.linalg.inv(T1.H))@picos.I(d)@T3)*Vp,((d**4),(Vp.shape[1])))
	Wp3 = picos.Constant("Wp3",(picos.I(d)@(numpy.linalg.inv(T1.H))@picos.I(d)@T4)*Vp,((d**4),(Vp.shape[1])))
	Wp4 = picos.Constant("Wp4",(picos.I(d)@(numpy.linalg.inv(T1.H))@picos.I(d)@T5)*Vp,((d**4),(Vp.shape[1])))

	# parameter matrix of S
	MatS1 = picos.HermitianVariable("MatS1",Vp.shape[1])
	MatS2 = picos.HermitianVariable("MatS2",Vp.shape[1])
	MatS3 = picos.HermitianVariable("MatS3",Vp.shape[1])
	MatS4 = picos.HermitianVariable("MatS4",Vp.shape[1])
	p = picos.RealVariable("p")

	ExtMat = picos.HermitianVariable("ExtMat",SymD*d**2)


	# SDP for PPT+MFLE relaxation of the success probability
	PPTprob = picos.Problem()
	PPTprob.set_objective("max",p)
	PPTprob.add_constraint(MatS1 >> 0)
	PPTprob.add_constraint((Wp1*MatS1*Wp1.H).partial_transpose((0,2)) >> 0)
	PPTprob.add_constraint(MatS2 >> 0)
	PPTprob.add_constraint((Wp2*MatS2*Wp2.H).partial_transpose((0,2)) >> 0)
	PPTprob.add_constraint(MatS3 >> 0)
	PPTprob.add_constraint((Wp3*MatS3*Wp3.H).partial_transpose((0,2)) >> 0)
	PPTprob.add_constraint(MatS4 >> 0)
	PPTprob.add_constraint((Wp4*MatS4*Wp4.H).partial_transpose((0,2)) >> 0)

	PPTprob.add_constraint(ExtMat >> 0)
	PPTprob.add_constraint(((VSYM@picos.I(d**2))*ExtMat*(VSYM@picos.I(d**2)).H).partial_transpose((0),(4,4,4)) >> 0)
	PPTprob.add_constraint(((VSYM@picos.I(d**2))*ExtMat*(VSYM@picos.I(d**2)).H).partial_transpose((2),(4,4,4)) >> 0)
	PPTprob.add_constraint(picos.I(d**4) >> Wp1*MatS1*Wp1.H + Wp2*MatS2*Wp2.H + Wp3*MatS3*Wp3.H + Wp4*MatS4*Wp4.H + ((picos.I(8)@SWAP@picos.I(2))*(VSYM@picos.I(d**2))*ExtMat*(VSYM@picos.I(d**2)).H*(picos.I(8)@SWAP@picos.I(2)).H).partial_trace((0),(4,4,4)))
	PPTprob.add_constraint(picos.I(d**4) << Wp1*MatS1*Wp1.H + Wp2*MatS2*Wp2.H + Wp3*MatS3*Wp3.H + Wp4*MatS4*Wp4.H + ((picos.I(8)@SWAP@picos.I(2))*(VSYM@picos.I(d**2))*ExtMat*(VSYM@picos.I(d**2)).H*(picos.I(8)@SWAP@picos.I(2)).H).partial_trace((0),(4,4,4)))

	#PPTprob.add_constraint(picos.I(d**4) - (Wp1*MatS1*Wp1.H + Wp2*MatS2*Wp2.H + Wp3*MatS3*Wp3.H + Wp4*MatS4*Wp4.H) >> 0)
	#PPTprob.add_constraint(picos.I(d**4) - (Wp1*MatS1*Wp1.H + Wp2*MatS2*Wp2.H + Wp3*MatS3*Wp3.H + Wp4*MatS4*Wp4.H).partial_transpose((0,2)) >> 0)

	PPTprob.add_constraint((tau @ picos.I(d**2)).H*(Wp1*MatS1*Wp1.H + Wp2*MatS2*Wp2.H + Wp3*MatS3*Wp3.H + Wp4*MatS4*Wp4.H)*(tau @ picos.I(d**2)) - p*picos.I(d**2) >> 0)

	PPTprob.solve(solver = "qics")

	print("(θ,MFLE)=", t1, PPTprob.value)
	return PPTprob.value

#################################
#PPT upper bound
#################################
def fPPT(t1):
	# resource entangled state (|τ>=I@T1*MES) R_AR_B
	tau = picos.Constant("τ",[numpy.cos(t1),0,0,numpy.sin(t1)],(4,1))
	T1 = picos.Constant("T1",[numpy.cos(t1),0,0,numpy.sin(t1)],(2,2))

	# PVM basis (|φi>=I@(T2)*MES)
	p1 = (numpy.sqrt(3)+1)/(2*numpy.sqrt(2))
	p2 = (numpy.sqrt(3)-1)/(2*numpy.sqrt(2))
	zeta = numpy.exp(1j*numpy.pi/3)
	T2 = picos.Constant("T2",[p1,0,0,-p2],(2,2))
	T3 = picos.Constant("T3",[p2,1,1,p1]/numpy.sqrt(3),(2,2))
	T4 = picos.Constant("T4",[p2,-zeta,zeta**2,p1]/numpy.sqrt(3),(2,2))
	T5 = picos.Constant("T5",[p2,zeta**2,-zeta,p1]/numpy.sqrt(3),(2,2))

	# isometry spanning the canonical subspace (R_A R_B A B)
	W1 = picos.Constant("W1",(picos.I(d)@(numpy.linalg.inv(T1.H))@picos.I(d)@T2)*V,((d**4),(V.shape[1])))
	W2 = picos.Constant("W2",(picos.I(d)@(numpy.linalg.inv(T1.H))@picos.I(d)@T3)*V,((d**4),(V.shape[1])))
	W3 = picos.Constant("W3",(picos.I(d)@(numpy.linalg.inv(T1.H))@picos.I(d)@T4)*V,((d**4),(V.shape[1])))
	W4 = picos.Constant("W4",(picos.I(d)@(numpy.linalg.inv(T1.H))@picos.I(d)@T5)*V,((d**4),(V.shape[1])))

	# parameter matrix of S
	MatS1 = picos.HermitianVariable("MatS1",V.shape[1])
	MatS2 = picos.HermitianVariable("MatS2",V.shape[1])
	MatS3 = picos.HermitianVariable("MatS3",V.shape[1])
	MatS4 = picos.HermitianVariable("MatS4",V.shape[1])

	p = picos.RealVariable("p")

	# SDP for PPT relaxation of the success probability
	PPTprob = picos.Problem()
	PPTprob.set_objective("max",p)
	PPTprob.add_constraint(MatS1 >> 0)
	PPTprob.add_constraint((W1*MatS1*W1.H).partial_transpose((0,2)) >> 0)
	PPTprob.add_constraint(MatS2 >> 0)
	PPTprob.add_constraint((W2*MatS2*W2.H).partial_transpose((0,2)) >> 0)
	PPTprob.add_constraint(MatS3 >> 0)
	PPTprob.add_constraint((W3*MatS3*W3.H).partial_transpose((0,2)) >> 0)
	PPTprob.add_constraint(MatS4 >> 0)
	PPTprob.add_constraint((W4*MatS4*W4.H).partial_transpose((0,2)) >> 0)

	PPTprob.add_constraint(picos.I(d**4) - (W1*MatS1*W1.H + W2*MatS2*W2.H + W3*MatS3*W3.H + W4*MatS4*W4.H) >> 0)
	PPTprob.add_constraint(picos.I(d**4) - (W1*MatS1*W1.H + W2*MatS2*W2.H + W3*MatS3*W3.H + W4*MatS4*W4.H).partial_transpose((0,2)) >> 0)

	PPTprob.add_constraint((tau @ picos.I(d**2)).H*(W1*MatS1*W1.H + W2*MatS2*W2.H + W3*MatS3*W3.H + W4*MatS4*W4.H)*(tau @ picos.I(d**2)) - p*picos.I(d**2) >> 0)

	PPTprob.solve(solver = "qics")

	print("(θ,PPT)=", t1, PPTprob.value)
	return PPTprob.value
	
#################################
#DPS 2nd Lv upper bound
#################################
def fDPS2(t1):
	# resource entangled state (|τ>=I@T1*MES) R_AR_B
	tau = picos.Constant("τ",[numpy.cos(t1),0,0,numpy.sin(t1)],(4,1))
	T1 = picos.Constant("T1",[numpy.cos(t1),0,0,numpy.sin(t1)],(2,2))

	# PVM basis (|φi>=I@(T2)*MES)
	p1 = (numpy.sqrt(3)+1)/(2*numpy.sqrt(2))
	p2 = (numpy.sqrt(3)-1)/(2*numpy.sqrt(2))
	zeta = numpy.exp(1j*numpy.pi/3)
	T2 = picos.Constant("T2",[p1,0,0,-p2],(2,2))
	T3 = picos.Constant("T3",[p2,1,1,p1]/numpy.sqrt(3),(2,2))
	T4 = picos.Constant("T4",[p2,-zeta,zeta**2,p1]/numpy.sqrt(3),(2,2))
	T5 = picos.Constant("T5",[p2,zeta**2,-zeta,p1]/numpy.sqrt(3),(2,2))

	# local linear transformation (A RA B RB)
	W1 = picos.Constant("W1",(picos.I(4)@T2@(numpy.linalg.inv(T1.H))),(d**4,d**4))
	W2 = picos.Constant("W2",(picos.I(4)@T3@(numpy.linalg.inv(T1.H))),(d**4,d**4))
	W3 = picos.Constant("W3",(picos.I(4)@T4@(numpy.linalg.inv(T1.H))),(d**4,d**4))
	W4 = picos.Constant("W4",(picos.I(4)@T5@(numpy.linalg.inv(T1.H))),(d**4,d**4))

	# parameter matrix of S ((W@picos.I(d**2))*S has range in R_A A R_B B R_B' B')
	MatS1 = picos.HermitianVariable("MatS1",28)
	MatS2 = picos.HermitianVariable("MatS2",28)
	MatS3 = picos.HermitianVariable("MatS3",28)
	MatS4 = picos.HermitianVariable("MatS4",28)

	ExtMat = picos.HermitianVariable("ExtMat",SymD*d**2)

	p = picos.RealVariable("p")

	# isometry for PPT operation (A'  RA'  A  RA  B  RB)
	VDPS = picos.Constant(
	[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,-(1.0/numpy.sqrt(2.0)),0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,-(1.0/numpy.sqrt(2.0)),0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-(1.0/numpy.sqrt(2.0)),0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0,0],[0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,-1.0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0],[0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,1.0,0,0,0,0,0,0,0,-(1.0/numpy.sqrt(2.0)),0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,-(1.0/numpy.sqrt(2.0)),0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0],[1.0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,-(1.0/numpy.sqrt(2.0)),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,-(1.0/numpy.sqrt(2.0)),0,0,0,-1.0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,-(1.0/numpy.sqrt(2.0)),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,-1.0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1.0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,-(1.0/numpy.sqrt(2.0)),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,1.0,0,0,0,0,0,0,0,-(1.0/numpy.sqrt(2.0)),0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,-(1.0/numpy.sqrt(2.0)),0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,-(1.0/numpy.sqrt(2.0)),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,-(1.0/numpy.sqrt(2.0)),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,-1.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1.0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,-(1.0/numpy.sqrt(2.0)),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,-(1.0/numpy.sqrt(2.0)),0,0,0,-1.0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,-(1.0/numpy.sqrt(2.0)),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,-1.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1.0/numpy.sqrt(2.0),0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,1.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,1.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,1.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

	# SDP for PPT relaxation of the success probability
	PPTprob = picos.Problem()
	PPTprob.set_objective("max",p)
	PPTprob.add_constraint(MatS1 >> 0)
	PPTprob.add_constraint((VDPS*MatS1*VDPS.H).partial_transpose((0),(4,4,4)) >> 0)
	PPTprob.add_constraint((VDPS*MatS1*VDPS.H).partial_transpose((2),(4,4,4)) >> 0)
	PPTprob.add_constraint(MatS2 >> 0)
	PPTprob.add_constraint((VDPS*MatS2*VDPS.H).partial_transpose((0),(4,4,4)) >> 0)
	PPTprob.add_constraint((VDPS*MatS2*VDPS.H).partial_transpose((2),(4,4,4)) >> 0)
	PPTprob.add_constraint(MatS3 >> 0)
	PPTprob.add_constraint((VDPS*MatS3*VDPS.H).partial_transpose((0),(4,4,4)) >> 0)
	PPTprob.add_constraint((VDPS*MatS3*VDPS.H).partial_transpose((2),(4,4,4)) >> 0)
	PPTprob.add_constraint(MatS4 >> 0)
	PPTprob.add_constraint((VDPS*MatS4*VDPS.H).partial_transpose((0),(4,4,4)) >> 0)
	PPTprob.add_constraint((VDPS*MatS4*VDPS.H).partial_transpose((2),(4,4,4)) >> 0)

	PPTprob.add_constraint(ExtMat >> 0)
	PPTprob.add_constraint(((VSYM@picos.I(d**2))*ExtMat*(VSYM@picos.I(d**2)).H).partial_transpose((0),(4,4,4)) >> 0)
	PPTprob.add_constraint(((VSYM@picos.I(d**2))*ExtMat*(VSYM@picos.I(d**2)).H).partial_transpose((2),(4,4,4)) >> 0)
	PPTprob.add_constraint(picos.I(d**4) >> W1*(VDPS*MatS1*VDPS.H).partial_trace((0),(4,4,4))*W1.H + W2*(VDPS*MatS2*VDPS.H).partial_trace((0),(4,4,4))*W2.H + W3*(VDPS*MatS3*VDPS.H).partial_trace((0),(4,4,4))*W3.H + W4*(VDPS*MatS4*VDPS.H).partial_trace((0),(4,4,4))*W4.H + ((VSYM@picos.I(d**2))*ExtMat*(VSYM@picos.I(d**2)).H).partial_trace((0),(4,4,4)))
	PPTprob.add_constraint(picos.I(d**4) << W1*(VDPS*MatS1*VDPS.H).partial_trace((0),(4,4,4))*W1.H + W2*(VDPS*MatS2*VDPS.H).partial_trace((0),(4,4,4))*W2.H + W3*(VDPS*MatS3*VDPS.H).partial_trace((0),(4,4,4))*W3.H + W4*(VDPS*MatS4*VDPS.H).partial_trace((0),(4,4,4))*W4.H + ((VSYM@picos.I(d**2))*ExtMat*(VSYM@picos.I(d**2)).H).partial_trace((0),(4,4,4)))

	PPTprob.add_constraint(0.5*(T2*T2.H).tr*(2.0*MatS1[0,0]+numpy.sqrt(2.0)*MatS1[0,5]+MatS1[4,4]+numpy.sqrt(2.0)*MatS1[4,8]+numpy.sqrt(2.0)*MatS1[5,0]+MatS1[5,5]+numpy.sqrt(2.0)*MatS1[8,4]+2.0*MatS1[8,8]+MatS1[10,10]+MatS1[10,15]+MatS1[14,14]-MatS1[14,21]+MatS1[15,10]+MatS1[15,15]-MatS1[21,14]+MatS1[21,21]) >= p)
	PPTprob.add_constraint(0.5*(T2*T2.H).tr*(2.0*MatS2[0,0]+numpy.sqrt(2.0)*MatS2[0,5]+MatS2[4,4]+numpy.sqrt(2.0)*MatS2[4,8]+numpy.sqrt(2.0)*MatS2[5,0]+MatS2[5,5]+numpy.sqrt(2.0)*MatS2[8,4]+2.0*MatS2[8,8]+MatS2[10,10]+MatS2[10,15]+MatS2[14,14]-MatS2[14,21]+MatS2[15,10]+MatS2[15,15]-MatS2[21,14]+MatS2[21,21]) >= p)
	PPTprob.add_constraint(0.5*(T2*T2.H).tr*(2.0*MatS3[0,0]+numpy.sqrt(2.0)*MatS3[0,5]+MatS3[4,4]+numpy.sqrt(2.0)*MatS3[4,8]+numpy.sqrt(2.0)*MatS3[5,0]+MatS3[5,5]+numpy.sqrt(2.0)*MatS3[8,4]+2.0*MatS3[8,8]+MatS3[10,10]+MatS3[10,15]+MatS3[14,14]-MatS3[14,21]+MatS3[15,10]+MatS3[15,15]-MatS3[21,14]+MatS3[21,21]) >= p)
	PPTprob.add_constraint(0.5*(T2*T2.H).tr*(2.0*MatS4[0,0]+numpy.sqrt(2.0)*MatS4[0,5]+MatS4[4,4]+numpy.sqrt(2.0)*MatS4[4,8]+numpy.sqrt(2.0)*MatS4[5,0]+MatS4[5,5]+numpy.sqrt(2.0)*MatS4[8,4]+2.0*MatS4[8,8]+MatS4[10,10]+MatS4[10,15]+MatS4[14,14]-MatS4[14,21]+MatS4[15,10]+MatS4[15,15]-MatS4[21,14]+MatS4[21,21]) >= p)

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

	# PVM basis (|φi>=I@(T2)*MES)
	p1 = (numpy.sqrt(3)+1)/(2*numpy.sqrt(2))
	p2 = (numpy.sqrt(3)-1)/(2*numpy.sqrt(2))
	zeta = numpy.exp(1j*numpy.pi/3)
	T2 = picos.Constant("T2",[p1,0,0,-p2],(2,2))
	T3 = picos.Constant("T3",[p2,1,1,p1]/numpy.sqrt(3),(2,2))
	T4 = picos.Constant("T4",[p2,-zeta,zeta**2,p1]/numpy.sqrt(3),(2,2))
	T5 = picos.Constant("T5",[p2,zeta**2,-zeta,p1]/numpy.sqrt(3),(2,2))

	#enet
	matA1 = [picos.Constant("A1[{}]".format(neti),numpy.random.randn(d**2,2).view(numpy.complex128),(d,d)) for neti in range(Enet)]	
	matBt1 = [picos.Constant("B1[{}]".format(neti),numpy.linalg.inv(matA1[neti]),(d,d)) for neti in range(Enet)]	
	matA2 = [picos.Constant("A2[{}]".format(neti),numpy.random.randn(d**2,2).view(numpy.complex128),(d,d)) for neti in range(Enet)]	
	matBt2 = [picos.Constant("B2[{}]".format(neti),numpy.linalg.inv(matA2[neti]),(d,d)) for neti in range(Enet)]	
	matA3 = [picos.Constant("A3[{}]".format(neti),numpy.random.randn(d**2,2).view(numpy.complex128),(d,d)) for neti in range(Enet)]	
	matBt3 = [picos.Constant("B3[{}]".format(neti),numpy.linalg.inv(matA3[neti]),(d,d)) for neti in range(Enet)]	
	matA4 = [picos.Constant("A4[{}]".format(neti),numpy.random.randn(d**2,2).view(numpy.complex128),(d,d)) for neti in range(Enet)]	
	matBt4 = [picos.Constant("B4[{}]".format(neti),numpy.linalg.inv(matA4[neti]),(d,d)) for neti in range(Enet)]	
	A = [picos.Constant("A[{}]".format(neti),numpy.random.randn(d**2,2).view(numpy.complex128),(d**2,1)) for neti in range(Enetc)]	
	RemAstate = [picos.Constant("RemA[{}]".format(neti),A[neti]*A[neti].H/(A[neti]*A[neti].H).tr,(d**2,d**2)) for neti in range(Enetc)]
	

	# vector in S∩V (R_A A R_B B)
	SVVec1 = [picos.Constant("Vec[{}]".format(neti),((picos.I(d)@matA1[neti])*MES)@(((numpy.linalg.inv(T1.H))@(T2*matBt1[neti].reshuffled("ji")))*MES),(d**4,1)) for neti in range(Enet)]
	SVVec2 = [picos.Constant("Vec[{}]".format(neti),((picos.I(d)@matA2[neti])*MES)@(((numpy.linalg.inv(T1.H))@(T3*matBt2[neti].reshuffled("ji")))*MES),(d**4,1)) for neti in range(Enet)]
	SVVec3 = [picos.Constant("Vec[{}]".format(neti),((picos.I(d)@matA3[neti])*MES)@(((numpy.linalg.inv(T1.H))@(T4*matBt3[neti].reshuffled("ji")))*MES),(d**4,1)) for neti in range(Enet)]
	SVVec4 = [picos.Constant("Vec[{}]".format(neti),((picos.I(d)@matA4[neti])*MES)@(((numpy.linalg.inv(T1.H))@(T5*matBt4[neti].reshuffled("ji")))*MES),(d**4,1)) for neti in range(Enet)]

	# variables for epsilon-net
	posvar = picos.RealVariable("posvar",Enet*4)
	p = picos.RealVariable("p")

	Dist = picos.RealVariable("δ")

	# SDP for computing the Schatten L1 norm between Butone and discretized SEP
	RemB = [picos.HermitianVariable("RemB[{}]".format(neti),d**2) for neti in range(Enetc)]
	trho = picos.HermitianVariable("tρ",d**4)
	
	# SDP for discretized SEP 
	SEPprob = picos.Problem()
	SEPprob.set_objective("max", p - Dist)
	#SEPprob.set_objective("max", picos.sum(posvar[neti] for neti in range(Enet*4))/4.0 - Dist)
	SEPprob.add_constraint(posvar >=0)
	SEPprob.add_constraint(Dist >= 0)
	SEPprob.add_constraint(Dist >= 2.0*trho.tr + picos.sum(RemB[neti].tr for neti in range(Enetc)) - d**4 + picos.sum(posvar[neti]*(SVVec1[neti]*SVVec1[neti].H).tr for neti in range(Enet)) + picos.sum(posvar[neti + Enet]*(SVVec2[neti]*SVVec2[neti].H).tr for neti in range(Enet)) + picos.sum(posvar[neti + Enet*2]*(SVVec3[neti]*SVVec3[neti].H).tr for neti in range(Enet)) + picos.sum(posvar[neti + Enet*3]*(SVVec4[neti]*SVVec4[neti].H).tr for neti in range(Enet)))

	SEPprob.add_constraint(trho >> 0)
	SEPprob.add_constraint(trho >> picos.I(d**4) - picos.sum(posvar[neti]*SVVec1[neti]*SVVec1[neti].H for neti in range(Enet)) - picos.sum(posvar[neti + Enet]*SVVec2[neti]*SVVec2[neti].H for neti in range(Enet)) - picos.sum(posvar[neti + Enet*2]*SVVec3[neti]*SVVec3[neti].H for neti in range(Enet)) - picos.sum(posvar[neti + Enet*3]*SVVec4[neti]*SVVec4[neti].H for neti in range(Enet)) - picos.sum(RemAstate[neti]@RemB[neti] for neti in range(Enetc)))
	SEPprob.add_list_of_constraints([RemB[neti] >> 0 for neti in range(Enetc)])

	SEPprob.add_constraint(picos.sum(posvar[neti] for neti in range(Enet)) >= p)
	SEPprob.add_constraint(picos.sum(posvar[neti + Enet] for neti in range(Enet)) >= p)
	SEPprob.add_constraint(picos.sum(posvar[neti + Enet*2] for neti in range(Enet)) >= p)
	SEPprob.add_constraint(picos.sum(posvar[neti + Enet*3] for neti in range(Enet)) >= p)

	SEPprob.options["rel_prim_fsb_tol"] = 3*10**(-5)
	SEPprob.options["rel_dual_fsb_tol"] = 3*10**(-5)
	SEPprob.options["rel_ipm_opt_tol"] = 1*10**(-8)
	
	try:
		SEPprob.solve(solver = "qics")
	except Exception as e:
		print(f"Error: {e}")
		return -1
	
	SEPBObound = p.value
	#SEPBObound = min([picos.sum(posvar[neti].value for neti in range(Enet)), picos.sum(posvar[neti+Enet].value for neti in range(Enet)), picos.sum(posvar[neti+Enet*2].value for neti in range(Enet)), picos.sum(posvar[neti+Enet*3].value for neti in range(Enet))])
	print("(θ,SEP but one)=", t1, SEPBObound)

	print("(θ,Distance between RemCP and SEP):", t1, Dist.value)	

	return SEPBObound/(1.0+Dist.value)

#################################
#MAIN
if __name__ == "__main__":
	t1 = [tstart + (tend-tstart)*step/div for step in range(div+1)]

	#with open('theta.txt', mode='w') as f:
    	#	f.write('\n'.join([str(t1[step]) for step in range(div+1)]))

	#MFLEstartTime = time.time()
	#with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
	#	futures = [executor.submit(fMFLE, t1[step]) for step in range(div+1)]
	#	MFLEbound = [f.result() for f in futures]
	#MFLEendTime = time.time()

	#print("PPT + MFLE time:",(MFLEendTime-MFLEstartTime)/(div+1))
	#with open('MFLE.txt', mode='w') as f:
    	#	f.write('\n'.join([str(MFLEbound[step]) for step in range(div+1)]))


	#PPTstartTime = time.time()
	#with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
	#	futures = [executor.submit(fPPT, t1[step]) for step in range(div+1)]
	#	PPTbound = [f.result() for f in futures]
	#PPTendTime = time.time()

	#print("PPT time:",(PPTendTime-PPTstartTime)/(div+1))
	#with open('PPT.txt', mode='w') as f:
    	#	f.write('\n'.join([str(PPTbound[step]) for step in range(div+1)]))


	#DPS2startTime = time.time()
	#with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
	#	futures = [executor.submit(fDPS2, t1[step]) for step in range(div+1)]
	#	DPS2bound = [f.result() for f in futures]
	#DPS2endTime = time.time()

	#print("DPS2 time:",(DPS2endTime-DPS2startTime)/(div+1))
	#with open('DPS2nd.txt', mode='w') as f:
    	#	f.write('\n'.join([str(DPS2bound[step]) for step in range(div+1)]))


	SEPstartTime = time.time()
	with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
		futures = [executor.submit(fSEP, t1[step]) for step in range(div+1)]
		SEPbound = [f.result() for f in futures]
	#SEPbound = [1 for f in range(div+1)]
	SEPendTime = time.time()

	print("SEP time:",(SEPendTime-SEPstartTime)/(div+1))
	with open('lowerbound.txt', mode='w') as f:
    		f.write('\n'.join([str(SEPbound[step]) for step in range(div+1)]))

	SEPboundc = []
	tc = []
	for step in range(div+1):
		if SEPbound[step]>=0:
			tc.append(t1[step])
			SEPboundc.append(SEPbound[step])


	#pyplot.title("")
	pyplot.xlabel("θ")
	pyplot.ylabel("success probability")
	#pyplot.plot(t1,PPTbound, label='PPT (DPS 1st Lv.)')
	#pyplot.plot(t1,DPS2bound, label='DPS 2nd Lv.')
	pyplot.plot(tc,SEPboundc, label='lower bound')
	#pyplot.plot(t1,MFLEbound, label='PPT + MFLE')
	pyplot.legend()
	pyplot.show()
