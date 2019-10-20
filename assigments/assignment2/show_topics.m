function show_topics(W, wl, fid)
% W is a V x k matrix, where V is the vocabulary size
% and k is number of topics. Note that each column of W
% should sum to one.

n_top_words = 6;

if nargin < 3
    fid = 1;
end
for i = 1:size(W, 2)
    w = W(:, i);
    [~, idx] = sort(w, 1, 'descend');
    top_words = sprintf('%s,', wl{idx(1:n_top_words)});
    
    fprintf(fid, 'W %d: %s\n', i, top_words);
end
end