function [out] = generateTrainingData(in)
%
% this function generates the training data
%

DEBUG = false;

if nargin==0
    in.Tmax = 2000; % size of the generated signal
    in.sigma = 0.1; %  variance of the additive noise
    in.sampling_rate = 1; % period samplingsign
    in.lambda = 0.1; % Source activity
    in.type = 'from_dictionary'; % 'from_dictionary' or 'random'
end

%% Initialize the parameters of the signal

% User-defined parameters
Tmax=in.Tmax;
sigma = in.sigma;
sampling_rate=in.sampling_rate;
lambda=in.lambda;

% support of pulse shapes and their parameters
t=(0:20);
gamma_params=(0.1:0.1:1);

%% Creation of a gamma-based shape dictionary
[~,shapes]= Create_DictionaryNHPP(sampling_rate,t,gamma_params,Tmax,'gamma','none',20);
nb_shapes=size(shapes,2);


%% Creation of the HPP points
st = 0;
while st==0
    [Tn]=CreateHPP(Tmax,lambda);
    st=length(Tn);
    
end

Tn(Tn+size(shapes,1)>Tmax)=[];

out.Tn = ceil(Tn);

%% Creation of the signal
[out.signal,out.energies]=Create_Signal_SyntheticNHPP(ceil(Tn*sampling_rate),Tmax,shapes,sigma,in.type);

end
