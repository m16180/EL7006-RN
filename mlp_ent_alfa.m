%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MATLAB function: mlp_ent_alfa.m
% Last update: 2/15/05, (C) Deniz Erdogmus, OGI School of Science and Engineering, Oregon,USA. 
% Modified by  Kyu-Hwa Jeong, CNEL, University of Florida, USA.
%
% PURPOSE: This program performs supervised batch training of an MLP.
%          Adaptation criterion is minimum sum of Renyi's error entropies for each
%          output channel,and pdf estimation is performed by Parzen windowing.
%          Steepest descent optimization is utilized
%
% FUNCTIONAL FORM:
% function [W1,W2,b1,b2,Vnorm,SIG] = mlp_ent_alfa(x,d,alfa,sig0,W10,W20,b10,kerneltype,eta,maxiter);
%
% Size of the weight matrix:
%         assuming the MLP has n0 inputs and n neurons in hidden layer W1=nxn0, b1=nx1,
%         assuming the MLP has nout output channels W2=nxnout,b2=noutx1
%         assuming linear output neurons
%         the output bias is set at the end of each iteration to yield zero mean error over training data set
%
% INPUT:
% x 		= training input vectors, each column is an input sample
% d 		= training target outputs, each column is the target output for corresponding input column
% alfa	    = Renyi's entropy order (allows alfa > 1 for now)
% sig 	    = size of the kernel in Parzen windowing
% W10		= initial weights of the first layer of MLP (assumed to have a single hidden layer)
% W20		= initial weights of the second layer of MLP
% b10		= initial bias values for hidden (first) layer
% kerneltype = 1 (Gaussian) is available now
% eta		= initial step size for steepest descent
% maxiter   = max # of iterations allowed
% 
%OUTPUT:
% W1        = weights of the first layer of MLP after training
% W2        = weights of the second layer of MLP after training
% b1        = bias of the first layer of MLP after training
% b2        = bias of the second layer of MLP after training
% Vnorm     = product of information potentials
% SIG       = Anealining Kernel size during training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [W1,W2,b1,b2,Vnorm,SIG] = mlp_ent_alfa(x,d,alfa,sig0,W10,W20,b10,kerneltype,eta,maxiter)

load pdfe_mse.mat;

N=length(d(1,:));			% # of training samples
n0 =length(W10(1,:));		% # of inputs
n = length(W10(:,1));		% # of hidden neurons
nout = length(d(:,1));		% # of outputs
W1=W10;W2=W20;b1=b10;		% initialize weights to specified values

iter = 0;
stopcriterion = (iter==maxiter);
if alfa > 1	% minimize entropy=maximize infpot
	while stopcriterion == 0
   	iter = iter + 1;
   
   	% evaluate error over training set
      z=W1*x+b1*ones(1,N);[y1,y1p]=nl(z);y2=W2'*y1;  e=d-y2;
      
      % set kernel size such that the estimated pdf has the same variance as the unbiased sample variance
%      sig = sig0*exp(-(iter-1)/(maxiter/3)+1)+0*(exp(-(iter-1)/(maxiter/3))+1)*N/(N-1)*(mean(e.^2,2)-mean(e,2).^2);	% find a suitable sig for each output channel
		if iter > 2, changesig=sum(abs(Vnorm(:,iter-1)-Vnorm(:,iter-2)))<0.002&sum(Vnorm(:,iter-1)>Vnorm(:,iter-2))>0; 
        else, changesig = 0; 
        end,
      if iter==1 
         sig = sig0;
		elseif iter>1 & changesig,
         sig = 0.99*sig;
         if sig < 0.8*sig0
            sig = 0.8*sig0;
         end
      end
		SIG(:,iter) = sig;
      for ccc = 1:nout
		 Vmaxpossible = infpot(zeros(1,N),alfa,sig(ccc),kerneltype);	% evaluate the max possible information potential
         V(ccc,iter) = infpot(e(ccc,:),alfa,sig(ccc),kerneltype);	% evaluate the information potential for each output
         Vnorm(ccc,iter) = V(ccc,iter)/Vmaxpossible;
         % the cost function is the sum of entropies (min) = product of information potentials (max)
      end
      
   	if (iter-1)/1==round((iter-1)/1),	% once in a while display progress
      	 figure(1);clf;
      	 subplot(2,2,1),plot(Vnorm');drawnow;	% plot normalized cost function for each output
         title('Learning curve(Information Potential)'); 
         subplot(2,2,2),plot(SIG');drawnow;
         title('Kernel size anealining');
         subplot(2,2,3),plot(d(1,:)-mean(d(1,:)),'r');hold on;plot(d(1,:)-e(1,:)-mean(d(1,:)-e(1,:)),'b');drawnow;
         title('system output and disired signal');
         subplot(2,2,4);
         [pdfx,pdfe]=pzpdf(e(1,:)-mean(e(1,:)));,plot(pdfx,pdfe,'b');drawnow;
         title('Probability densities of errors');
         hold on;
         plot(pdfx,pdfe_mse,'r');drawnow;
         legend('MEE','MSE');
         f(iter)=getframe(figure(1));
        
      end
   
   	% evaluate gradient at current weight vector
      [dVdW1,dVdW2,dVdb1] = dvdwm(x,y1p,y1,e,W2,sig,alfa,kerneltype);
      
      % adjust stepsize
      W1t=W1+eta*dVdW1;W2t=W2+eta*dVdW2;b1t=b1+eta*dVdb1;
      zt=W1t*x+b1t*ones(1,N);[y1t,y1pt]=nl(zt);y2t=W2t'*y1t;et=d-y2t;%sigt=N/(N-1)*(mean(et.^2,2)-mean(et,2).^2);
      for ccc = 1:nout,
%         Vmaxpossible = infpot(zeros(1,N),alfa,sigt(ccc),kerneltype);
         Vt=infpot(et(ccc,:),alfa,sig(ccc),kerneltype);
      	Vnormt(ccc)=Vt/Vmaxpossible;   
      end
      if prod(Vnormt)>prod(Vnorm(:,iter)),W1=W1t;W2=W2t;b1=b1t;eta=1.1*eta,
      elseif prod(Vnormt)<=prod(Vnorm(:,iter)),eta=0.7*eta,
      end,
   
   	stopcriterion = (iter==maxiter);
	end	% while stopcriterion
	% adjust output bias to make mean of error = zero
	z=W1*x+b1*ones(1,N);[y1,y1p]=nl(z);y2=W2'*y1;e=d-y2;b2=mean(e,2);
    sig=sig0*exp(-iter/(maxiter/3))+N/(N-1)*(mean(e.^2,2)-mean(e,2).^2);   
   SIG(:,iter+1)=sig;
   for ccc = 1:nout,
      Vmaxpossible = infpot(zeros(1,N),alfa,sig(ccc),kerneltype);
      V=infpot(e(ccc,:),alfa,sig(ccc),kerneltype);
      Vnorm(ccc,iter+1)=V/Vmaxpossible;
   end,
elseif alfa < 1 %minimize entropy=minimize infpot
end
%movie2avi(f,'result_tmp');
movie2avi(f,'result_tmp','fps',10,'quality',95);
%keyboard;
%return;
%%%%%%%% SUBROUTINES %%%%%%%%
function V = infpot(e,alfa,sig,kerneltype)
% evaluates the information potential for the given samples of e
N=length(e);	% number of samples
V=0;
for n=1:N,
   V = V + sum(kernel(e(n)-e,sig,kerneltype))^(alfa-1);
end,
V = V/N^alfa;
%%%%%%%%
function [dVdW1m,dVdW2m,dVdb1m] = dvdwm(x,y1p,y1,e,W2,sig,alfa,kerneltype);
% evaluates the overall gradient combining contributions from all outputs
nout = length(W2(1,:));N = length(x(1,:));n = length(W2(:,1));n0 = length(x(:,1));
if nout == 1	% if there is a single output, then gradient is directly equal to its contribution
   [dVdW1m,dVdW2m,dVdb1m] = dvdw(x,y1p,y1,e,W2,sig,alfa,kerneltype);
elseif nout > 1
   for ccc = 1:nout
		for j = 1:N
			for i = 1:N
   			ker(j,i) = kernel(e(ccc,j)-e(ccc,i),sig(ccc),kerneltype);
  			end      
         t1(j,1) = (sum(ker(j,:))^(alfa-1));
      end    
      Vo(ccc,1) = sum(t1)/N^alfa;
   end
   Vop = ones(nout,1);
   for ccc = 1:nout
      for ccc2 = 1:nout
         if ccc~=ccc2
            Vop(ccc) = Vop(ccc)*Vo(ccc2);
         end
      end
   end
   dVdW2m = zeros(n,nout);dVdW1m = zeros(n,n0);dVdb1m = zeros(n,1);
   for ccc = 1:nout
      [dVdW1,dVdW2,dVdb1] = dvdw(x,y1p,y1,e(ccc,:),W2(:,ccc),sig(ccc),alfa,kerneltype);
      dVdW2m(:,ccc) = dVdW2m(:,ccc) + Vop(ccc)*dVdW2;
      dVdW1m = dVdW1m + Vop(ccc)*dVdW1;      
      dVdb1m = dVdb1m + Vop(ccc)*dVdb1;
	end
end
%%%%%%%%
function [dVdW1,dVdW2,dVdb1] = dvdw(x,y1p,y1,e,W2,sig,alfa,kerneltype)
% evaluates the gradient contribution of a single output
% y1p : derivative of hidden layer outputs(after nonlinearity)
% y1 : hidden layer outputs(after nonlinearity)
% obtain the derivative of NN output wrt weights
[dy2dW1,dy2db1,dy2dW2,dy2db2] = dydw(x,y1p,y1,W2);
N = length(x(1,:));n0 = length(x(:,1));n = length(W2(:,1));
dVdW2 = zeros(n,1);dVdW1 = zeros(n,n0);dVdb1 = zeros(n,1);
for j = 1:N
	for i = 1:N
   	ker(j,i) = kernel(e(1,j)-e(1,i),sig,kerneltype);
  	end      
  	t1(j,1) = (sum(ker(j,:))^(alfa-2));
  	dVdW2t = zeros(n,1);dVdW1t = zeros(n,n0);dVdb1t = zeros(n,1);
  	for i = 1:N
     	t2 = kernelpr(e(1,j)-e(1,i),sig,kerneltype);
   	dVdW2t = dVdW2t + t2*(dy2dW2(:,i)-dy2dW2(:,j));
   	dVdW1t = dVdW1t + t2*(dy2dW1(:,:,i)-dy2dW1(:,:,j));
   	dVdb1t = dVdb1t + t2*(dy2db1(:,i)-dy2db1(:,j));
  	end
	dVdW2=dVdW2+t1(j,1)*dVdW2t;dVdW1=dVdW1+t1(j,1)*dVdW1t;dVdb1=dVdb1+t1(j,1)*dVdb1t;
end
dVdW2 = (alfa-1)/(N^alfa)*dVdW2;dVdW1 = (alfa-1)/(N^alfa)*dVdW1;dVdb1 = (alfa-1)/(N^alfa)*dVdb1;
% dVdb2 = 0 for entropy criterion since mean of pdf does not affect the entropy
%%%%%%%%
function [dy2dW1,dy2db1,dy2dW2,dy2db2] = dydw(x,y1p,y1,W2)
% evaluates the gradient of mlp output wrt the weights
N = length(x(1,:));n0 = length(x(:,1));
dy2db2 = ones(1,N);
for n = 1:N
   dy2dW2(:,n) = y1(:,n);
  	dy2db1(:,n) = W2.*y1p(:,n);
	for j = 1:n0
   	dy2dW1(:,j,n) = dy2db1(:,n)*x(j,n);
   end
end
%%%%%%%%
function out = kernel(in,sig,kerneltype)
% evaluates the kernel for the given input
% accepts 1-D inputs only,
if kerneltype == 1	% Gaussian
	in = in(:);in=in';out = (1/sqrt(2*pi*sig^2))*exp(-0.5*(in.^2)/sig^2);
end
%%%%%%%%
function out = kernelpr(in,sig,kerneltype)
% evaluates the derivative of the kernel at the given points
% accepts 1-D inputs only,
if kerneltype == 1	% Gaussian
	in = in(:);in=in';out = -in.*kernel(in,sig,kerneltype)/sig^2;
end
%%%%%%%%
function [y,yp] = nl(x)
% nonlinearity of neurons
% y = f(x), yp = f'(x)
y=tanh(x);yp=ones(size(y))-y.^2;	% d(tanh(x))/dx=1-tanh(x)^2

% ALPHA=10; 
% y=1./(1+exp(-ALPHA*x));
% yp=ALPHA*y.*(1-y);



%%%%%%%%
