classdef supLearn
% describes a generic supervised learning class
 methods (Abstract)
   %learner = init(varargin);
	 learner = train(learner, Xtrain, Ytrain);
	 Ytest   = predict(learner, Xtest);
  end
end

