function [Pt]=CreateHPP(T,lamb)

Pt = []; % Set of event times.
S_last = 0; % Running sum of interarrival times

while (S_last < T),
    S_last = S_last - (1/lamb)*log(rand); % add an extra event
    Pt = [Pt, S_last];
end

Pt=Pt(1:length(Pt)-1);