function [pile_up,energies]=Create_Signal_SyntheticNHPP(T,signal_size,shapes,sigma,type)

%% Draws the energies based on bimodal gaussian mixture
seed = rand(length(T),1);
energies = (seed < 0.8) .* (10*randn(length(T),1)+100) + (seed > 0.8) .* (20*randn(length(T),1)+225);
energies = energies';

if strcmpi(type,'from_dictionary')
    pile_up=zeros(signal_size,1);
    shapechoice=unidrnd(size(shapes,2),1,length(T));
    for k=1:length(T)
        pulse_shape= shapes(:,shapechoice(k));
        pulse_shape = pulse_shape / max(pulse_shape);
        pulse_shape = pulse_shape * energies(k);
        if T(k)+length(pulse_shape)-1 < signal_size
            pile_up(T(k):T(k)+length(pulse_shape)-1)=pile_up(T(k):T(k)+length(pulse_shape)-1)+pulse_shape;
        end
    end
    pile_up=pile_up+sigma*randn(signal_size,1);
elseif strcmpi(type,'random')
    pile_up=zeros(signal_size,1);
    for k=1:length(T)
        temp=(0:size(shapes,2)-1);
        pulse_shape=(temp'*(rand(1,1)+0.3)).*exp(-temp'*(rand(1,1)+0.3)) ;
        pulse_shape = pulse_shape / max(pulse_shape);
        pulse_shape = pulse_shape*energies(k);
        pile_up(T(k):T(k)+length(temp)-1)=pile_up(T(k):T(k)+length(temp)-1)+pulse_shape;
    end
    pile_up=pile_up+sigma*randn(signal_size,1);
end

pile_up = pile_up';


end
