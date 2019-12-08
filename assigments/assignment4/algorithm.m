function prob = algorithm(q)

% plot and return the probability
    %% plot price movements
    load('sp500.mat');
    figure;
    yyaxis left
    plot(price_move);
    xlabel('Week');
    ylabel('Up or Down')
    % note that for HERE the y is the observation 
    % while x is the hidden variable
    % we are looking for Prob(x|y,M) -- Decoding problem
    %% Forward-backward to get posterior decoding
    conP=[[q,1-q];[1-q,q]];
    nrow=size(price_move,1);
    %% forward, like problem 3
    % initialization
    a = [[0.8,1-0.8];[1-0.8,0.8]];
    alpha=zeros(nrow,2);  % row: trail; col: 1 for good, 2 for bad
    % when price_move(t) = +1 -> up
    % (3 - price_move(1))) / 2 = 1
    % conP(k, good)
    % when price_move(t) = -1 -> down
    % (3 - price_move(1))) / 2 = 2
    % conP(k, bad)
    alpha(1,1)=conP(1,(3-price_move(1))/2) * 0.2;
    alpha(1,2)=conP(2,(3-price_move(1))/2) * 0.8;
    for t = 2:nrow
        alpha(t,1)=conP(1,(3-price_move(t))/2)*(alpha(t-1,1)*a(1,1)+alpha(t-1,2)*a(2,1));
        alpha(t,2)=conP(2,(3-price_move(t))/2)*(alpha(t-1,1)*a(1,2)+alpha(t-1,2)*a(2,2));
    end
    px=alpha(nrow,1)+alpha(nrow,2);
    %% backward
    beta=zeros(nrow,2);
    beta(nrow,1)=1;
    beta(nrow,2)=1;
    for t = nrow-1:-1:1
        beta(t,1)=a(1,1)*conP(1,(3-price_move(t+1))/2)*beta(t+1,1)+a(2,1)*conP(2,(3-price_move(t+1))/2)*beta(t+1,2);
        beta(t,2)=a(2,1)*conP(1,(3-price_move(t+1))/2)*beta(t+1,1)+a(2,2)*conP(2,(3-price_move(t+1))/2)*beta(t+1,2);
    end
    %% posterior
    postP=alpha.*beta/px;
    %% plot in figure
    yyaxis right
    plot(postP(:,1));
    ylabel('P(x=good|y)')
    title(sprintf('Plots with q = %.2f',q));
    %% output the result of week 39
    prob=postP(39,1);

end