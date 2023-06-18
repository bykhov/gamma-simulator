clearvars
clc

%% Lambda scale
lambda_to_train = 0.05:0.1:0.65;
lambda_to_test = 0.05:0.05:0.65;

N_test = 1100;
N_train = 2000;

in.Tmax = 1024; % size of the generated signal
in.sigma = 0.1; %  variance of the additive noise
in.sampling_rate = 1; % period samplingsign
in.type = 'from_dictionary'; % 'from_dictionary' or 'random'

%% Train database
X_train_all = [];
Y_train_all = [];

for n=1:numel(lambda_to_train)
    in.lambda = lambda_to_train(n); % Source activity

    X_train = zeros(N_train,in.Tmax);
    Y_train = zeros(size(X_train));


    parfor k = 1:N_train
        frame = generateTrainingData(in);
        X_train(k,:) = frame.signal;
        Y_train(k,:) = conv_to_int(frame);
        if mod(k,500) == 0
            disp(k)
        end
    end

    % currentSavename = ['data_train_' num2str(100*in.lambda) '.mat'];
    %save(currentSavename,'X_train','Y_train');
    histcounts(Y_train(:),0:8)
    [r,~] = find(Y_train >= 6); % up to 5 particles per one sample!!!
    disp(['number of 6 or more: ',num2str(length(r))])
    Y_train(r,:) = [];
    X_train(r,:) = [];
    X_train_all = [X_train_all; X_train];
    Y_train_all = [Y_train_all; Y_train];

    disp(100*in.lambda)
end
save('train',"X_train_all","Y_train_all")

%% Test database

for n=1:numel(lambda_to_test)

    in.lambda = lambda_to_test(n); % Source activity

    X_test = zeros(N_test,in.Tmax);
    Y_test = zeros(size(X_test));

    parfor k = 1:N_test
        frame = generateTrainingData(in);
        X_test(k,:) = frame.signal;
        Y_test(k,:) = conv_to_int(frame);
    end
    histcounts(Y_test(:),0:8)

    [r,~] = find(Y_test >= 6); % up to 5 particles per one sample!!!
    disp(['number of 6 or more: ',num2str(length(r))])
    X_test(r,:) = [];
    Y_test(r,:) = [];

    currentSavename = ['data_test_' num2str(100*in.lambda) '.mat'];
    save(currentSavename,'X_test','Y_test');
    disp(['lambda = ' num2str(in.lambda,3)])
end

%%
%histogram(Y_train_all(:))
%%
%Convert Tn to binary
function Y = conv_to_int(frame)
    Y = zeros(1,length(frame.signal));
    idx = round(frame.Tn) + 1;
    for k = idx   
        Y(k) = Y(k) + 1;
    end
end

