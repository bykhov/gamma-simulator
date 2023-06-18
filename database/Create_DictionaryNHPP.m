function [A,shapes]= Create_DictionaryNHPP(samplingrate,t,lambda,Tmax,shape,shape_reset,rangesave)


nb_words=(length(lambda))*(ceil(Tmax*samplingrate)-length(t));
shapes=zeros(max(length(t),length(shape_reset)),length(lambda));

switch lower(shape),
    case {'exponential'}
        shapes(1:length(t),1:length(lambda))=exp(-t'*lambda);
    case {'gamma'}
        shapes(1:length(t),1:length(lambda))=(t'*(lambda+0.3)).*exp(-t'*(lambda+0.3)) ; % il n'y aurait pas un probleme la? c'est t puissance normalement
        shapes(1:length(t),1:length(lambda))=shapes(1:length(t),1:length(lambda))./repmat(sum(shapes(1:length(t),1:length(lambda))),length(t),1);
    otherwise
        error('Enter the shape of the pulse !');
end

try
A=zeros(ceil(Tmax*samplingrate)+rangesave,nb_words);
for k=1:ceil(samplingrate*Tmax)+rangesave-length(t),
    A(k:k+size(shapes,1)-1 , (k-1)* (size(shapes,2)) +1:k*size(shapes,2))=shapes;
end
catch
    A=[];
end