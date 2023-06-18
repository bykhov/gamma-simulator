clearvars
clc

%% Lambda scale
lambda_to_train = 0.05:0.1:0.65;
lambda_to_test = 0.1:0.1:0.6;

N_test = 1100;
N_train = 2000;

in.Tmax = 1024; % size of the generated signal
in.sigma = 0.1; %  variance of the additive noise
in.sampling_rate = 1; % period samplingsign
in.type = 'from_dictionary'; % 'from_dictionary' or 'random'


stat = zeros(1,6);

for n=1:numel(lambda_to_test)
    in.lambda = lambda_to_test(n); % Source activity

    currentSavename = ['data_test_' num2str(100*in.lambda) '.mat'];
    load(currentSavename,'Y_test');
    stat = stat + sum(Y_test(:) == [0:5]);

end
weights = max(stat)./stat;
save('data_test_weights',weights)