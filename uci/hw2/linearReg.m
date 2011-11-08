classdef linearReg < supLearn
% Class implementing linear regression
  properties (SetAccess=private, GetAccess=private)
		theta = [];
	end;
	methods
		% Constructor (takes no arguments or training data)
		function obj = linearReg(Xtr,Ytr, varargin)
			if (nargin > 0) 
				obj=train(obj,Xtr,Ytr, varargin{:});
			end;
		end

		% Batch, closed form solution
		function obj=train(obj, Xtr,Ytr, reg)
		  if (nargin < 4)
				obj.theta = (Xtr\Ytr)';
			else
				[M,N]=size(Xtr);
				obj.theta = Ytr'*(Xtr/M)*inv(Xtr'*Xtr/M + reg*eye(N));
			end;
		end

		% Test function: predict on Xtest
		function Yte = predict(obj,Xte)
			Yte = Xte * obj.theta';
		end

		% display function, print out coefficients
		function display(obj)
			display(obj.theta);
		end

%
% Evaluation / cost functions
%
		% calculate mean squared error for a given validation data set
		function err = mse(obj,Xval,Yval)
			Yhat = obj.predict(Xval);
			err = mean( (Yhat-Yval).^2 );
		end

		% calculate mean absolute error for a given validation data set
		function err = mae(obj,Xval,Yval)
			Yhat = obj.predict(Xval);
			err = mean( abs(Yhat-Yval) );
		end

  end
end
