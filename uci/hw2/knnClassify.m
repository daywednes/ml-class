classdef knnClassify < supLearn
% Class implementing K-nearest-neighbors classifier
  properties (SetAccess=private, GetAccess=private)
		K=1;
		Xtrain=[];	% Training features, Ndata x Nfeatures
		Ytrain=[];	% Training classes, Ndata x 1
	end;
	methods
		% Constructor (takes zero arguments or 3)
		function obj = knnClassify(K,Xtr,Ytr)
			if (nargin > 0) 
				obj.K = K;
				obj.Xtrain = Xtr;
				obj.Ytrain = Ytr;
			end;
		end

		% set parameter K if desired
		function obj=setK(obj, K)
			obj.K = K;
		end

		% Batch training: just memorize data
		function obj=train(obj, Xtr,Ytr)
			obj.Xtrain = Xtr;
			obj.Ytrain = Ytr;
		end

		% Test function: predict on Xtest
		function Yte = predict(obj,Xte)
			[Ntr,Mtr] = size(obj.Xtrain);					% get size of training, test data
			[Nte,Mte] = size(Xte);
			classes = unique(obj.Ytrain);					% figure out how many classes & their labels
			Yte = repmat(obj.Ytrain(1), [Nte,1]);	% make Ytest the same data type as Ytrain
			K = min(obj.K, Ntr);									% can't have more than Ntrain neighbors
			for i=1:Nte,													% For each test example:
				dist = sum( bsxfun( @minus, obj.Xtrain, Xte(i,:) ).^2 , 2);	% compute sum of squared differences
				[tmp,idx] = sort(dist);							% find nearest neighbors over Xtrain (dimension 2)
				cMax=1; NcMax=0;										% then find the majority class within that set of neighbors
				for c=1:length(classes),
					Nc = sum(obj.Ytrain(idx(1:K))==classes(c));	% count up how many instances of that class we have
					if (Nc>NcMax), cMax=c; NcMax=Nc; end;				% save the largest count and its class id
				end;
				Yte(i)=classes(cMax);								% save results
			end;
		end

end	% end methods
end % end class
