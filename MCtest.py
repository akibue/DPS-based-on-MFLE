import concurrent.futures
import os

import numpy
import picos
from matplotlib import pyplot
import time

# local dimension of the canonical subspace
d = 5

#################################START
#Preparation for canonical subspace
# unifV
_unifV = numpy.zeros((d,1))
for i in range(d):
	_unifV[i] = 1.0/numpy.sqrt(d)
unifV = picos.Constant("unifV",_unifV,(d,1))

# isometry spanning unifV complement subspace
_VMESC = numpy.zeros((d,d-1),dtype=numpy.complex128)
for i in range(d):
	for j in range(1,d):
		_VMESC[i,j-1] = numpy.exp(2j*numpy.pi*i*j/d)/numpy.sqrt(d)
VMESC = picos.Constant("VMESC",_VMESC,(d,d-1))

# CNOT isometry
_CNOT = numpy.zeros((d**2,d))
for i in range(d):
	_CNOT[i*d+i,i] = 1.0
CNOT = picos.Constant("CNOT",_CNOT,(d**2,d))
#################################END

# isometry spanning the canonical subspace and MFLE (R AB)
Vp = picos.Constant("Vp",(unifV@unifV & VMESC@VMESC),(d**2,d**2-2*(d-1)))
V = picos.Constant("V",(unifV@unifV & VMESC@picos.I(d)),(d**2,d**2-(d-1)))


#################################
#INITIALIZATION
#################################START
tstart = -3.0
tend = 0.0
div = 23

#################################END

def Diag2(index):
	_DMat = numpy.zeros((d**2,d**2))
	d1 = index // d
	d2 = index % d
	if d1==d2:
		return _DMat
	else:
		_DMat[d1*d+d2,d1*d+d2] = 1.0
		return _DMat

def Diag4(index):
	_DMat = numpy.zeros((d**4,d**4))
	d3 = index // d
	d4 = index % d
	d2 = d3 // d
	d3 = d3 % d
	d1 = d2 // d
	d2 = d2 % d
	if (d1==d2) or (d3==d4):
		return _DMat
	else:
		_DMat[d1*d**3+d2*d**2+d3*d+d4,d1*d**3+d2*d**2+d3*d+d4] = 1.0
		return _DMat


#################################
#MFLE upper bound
#################################
def fMFLE(t1):
	# resource entangled state (|τ>=I@T1*MES) R_AR_B
	zeta = numpy.sum(numpy.exp(t1*numpy.arange(d)))
	_T1 = numpy.zeros((d,d))
	for i in range(d):
		_T1[i,i] = numpy.sqrt(numpy.exp(t1*i)/zeta)
	T1 = picos.Constant("T1",_T1,(d,d))

	# target state (|φ>=I@(T2)*MES) AB
	t2 = -2.63392
	zeta = numpy.sum(numpy.exp(t2*numpy.arange(d)))
	_T2 = numpy.zeros((d,d))
	for i in range(d):
		_T2[i,i] = numpy.sqrt(numpy.exp(t2*i)/zeta)
	T2 = picos.Constant("T2",_T2,(d,d))
	
	# isometry spanning the canonical subspace (R AB)
	Wp = picos.Constant("Wp",(CNOT@CNOT)*((numpy.linalg.inv(T1.H))@T2)*Vp,(d**4,Vp.shape[1]))
	# isometry spanning the canonical subspace (R)
	W2 = picos.Constant("W2",CNOT*(numpy.linalg.inv(T1.H))*VMESC,(d**2,VMESC.shape[1]))
	# isometry spanning the canonical subspace (AB)
	W3 = picos.Constant("W3",CNOT*T2*VMESC,(d**2,VMESC.shape[1]))

	# parameter matrices
	S1 = picos.HermitianVariable("S1",Vp.shape[1])
	S2 = [picos.HermitianVariable("S2[{}]".format(neti),d-1) for neti in range(d**2)]
	S3 = [picos.HermitianVariable("S3[{}]".format(neti),d-1) for neti in range(d**2)]
	svar = picos.RealVariable("svar",d**4)
	

	# SDP for PPT+MFLE relaxation of the success probability
	PPTprob = picos.Problem()
	PPTprob.set_objective("max",S1[0,0])
	PPTprob.add_constraint(svar >=0)
	PPTprob.add_constraint(S1 >> 0)
	PPTprob.add_list_of_constraints([S2[neti] >> 0 for neti in range(d**2)])
	PPTprob.add_list_of_constraints([S3[neti] >> 0 for neti in range(d**2)])
	#R_A R_B A B
	S = Wp*S1*Wp.H + picos.sum((W2*S2[neti]*W2.H)@Diag2(neti) for neti in range(d**2)) + picos.sum(Diag2(neti)@(W3*S3[neti]*W3.H) for neti in range(d**2)) + picos.sum(svar[neti]*Diag4(neti) for neti in range(d**4))  
	PPTprob.add_constraint(S.partial_transpose((0,2),(d,d,d,d)) >> 0)
	PPTprob.add_constraint(picos.I(d**4) - S >> 0)
	PPTprob.add_constraint(picos.I(d**4) - S.partial_transpose((0,2),(d,d,d,d)) >> 0)

	PPTprob.solve(solver = "qics")

	print("(θ,MFLE)=", t1, PPTprob.value)
	return PPTprob.value

#################################
#PPT upper bound
#################################
def fPPT(t1):
	# resource entangled state (|τ>=I@T1*MES) R_AR_B
	zeta = numpy.sum(numpy.exp(t1*numpy.arange(d)))
	_T1 = numpy.zeros((d,d))
	for i in range(d):
		_T1[i,i] = numpy.sqrt(numpy.exp(t1*i)/zeta)
	T1 = picos.Constant("T1",_T1,(d,d))

	# target state (|φ>=I@(T2)*MES) AB
	t2 = -2.63392
	zeta = numpy.sum(numpy.exp(t2*numpy.arange(d)))
	_T2 = numpy.zeros((d,d))
	for i in range(d):
		_T2[i,i] = numpy.sqrt(numpy.exp(t2*i)/zeta)
	T2 = picos.Constant("T2",_T2,(d,d))
	
	# isometry spanning the canonical subspace (R AB)
	W = picos.Constant("W",(CNOT@CNOT)*((numpy.linalg.inv(T1.H))@T2)*V,(d**4,V.shape[1]))
	# isometry spanning the canonical subspace (R)
	W2 = picos.Constant("W2",CNOT*(numpy.linalg.inv(T1.H))*VMESC,(d**2,VMESC.shape[1]))
	# isometry spanning the canonical subspace (AB)
	W3 = picos.Constant("W3",CNOT*T2,(d**2,d))

	# parameter matrices
	S1 = picos.HermitianVariable("S1",V.shape[1])
	S2 = [picos.HermitianVariable("S2[{}]".format(neti),d-1) for neti in range(d**2)]
	S3 = [picos.HermitianVariable("S3[{}]".format(neti),d) for neti in range(d**2)]
	svar = picos.RealVariable("svar",d**4)
	

	# SDP for PPT+MFLE relaxation of the success probability
	PPTprob = picos.Problem()
	PPTprob.set_objective("max",S1[0,0])
	PPTprob.add_constraint(svar >=0)
	PPTprob.add_constraint(S1 >> 0)
	PPTprob.add_list_of_constraints([S2[neti] >> 0 for neti in range(d**2)])
	PPTprob.add_list_of_constraints([S3[neti] >> 0 for neti in range(d**2)])
	#R_A R_B A B
	S = W*S1*W.H + picos.sum((W2*S2[neti]*W2.H)@Diag2(neti) for neti in range(d**2)) + picos.sum(Diag2(neti)@(W3*S3[neti]*W3.H) for neti in range(d**2)) + picos.sum(svar[neti]*Diag4(neti) for neti in range(d**4))  
	PPTprob.add_constraint(S.partial_transpose((0,2),(d,d,d,d)) >> 0)
	PPTprob.add_constraint(picos.I(d**4) - S >> 0)
	PPTprob.add_constraint(picos.I(d**4) - S.partial_transpose((0,2),(d,d,d,d)) >> 0)

	PPTprob.solve(solver = "qics")

	print("(θ,PPT)=", t1, PPTprob.value)
	return PPTprob.value
	
#MAIN
if __name__ == "__main__":
	t1 = [tstart + (tend-tstart)*step/div for step in range(div+1)]
	MFLEstartTime = time.time()
	with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
		futures = [executor.submit(fMFLE, t1[step]) for step in range(div+1)]
		MFLEbound = [f.result() for f in futures]
	MFLEendTime = time.time()

	PPTstartTime = time.time()
	with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
		futures = [executor.submit(fPPT, t1[step]) for step in range(div+1)]
		PPTbound = [f.result() for f in futures]
	PPTendTime = time.time()


	print("PPT + MFLE time:",(MFLEendTime-MFLEstartTime)/(div+1))
	print("PPT time:",(PPTendTime-PPTstartTime)/(div+1))


	with open('t_5d.txt', mode='w') as f:
    		f.write('\n'.join([str(t1[step]) for step in range(div+1)]))

	with open('MFLE_5d.txt', mode='w') as f:
    		f.write('\n'.join([str(MFLEbound[step]) for step in range(div+1)]))

	with open('PPT_5d.txt', mode='w') as f:
    		f.write('\n'.join([str(PPTbound[step]) for step in range(div+1)]))

	
	#pyplot.title("")
	pyplot.xlabel("t")
	pyplot.ylabel("q(φ,τ)")
	pyplot.plot(t1,PPTbound, label='PPT (DPS 1st Lv.)')
	pyplot.plot(t1,MFLEbound, color='crimson', label='PPT + MFLE')
	pyplot.legend()
	pyplot.show()
