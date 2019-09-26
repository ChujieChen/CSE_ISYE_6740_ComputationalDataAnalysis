function [ class, centroid ] = mykmeans( pixels, K )
%
% Your goal of this assignment is implementing your own K-means.
%
% Input:
%     pixels: data set. Each row contains one data point. For image
%     dataset, it contains 3 columns, each column corresponding to Red,
%     Green, and Blue component.
%
%     K: the number of desired clusters. Too high value of K may result in
%     empty cluster error. Then, you need to reduce it.
%
% Output:
%     class: the class assignment of each data point in pixels. The
%     assignment should be 1, 2, 3, etc. For K = 5, for example, each cell
%     of class should be either 1, 2, 3, 4, or 5. The output should be a
%     column vector with size(pixels, 1) elements.
%
%     centroid: the location of K centroids in your result. With images,
%     each centroid corresponds to the representative color of each
%     cluster. The output should be a matrix with size(pixels, 1) rows and
%     3 columns. The range of values should be [0, 255].
%     
%
% You may run the following line, then you can see what should be done.
% For submission, you need to code your own implementation without using
% the kmeans matlab function directly. That is, you need to comment it out.

% 	[class, centroid] = kmeans(pixels, K);
%         [class, centroid] = kmeans(pixels, K, "Distance","cosine");
%         [class, centroid] = kmedoids(pixels, K, "Distance","cityblock");

%%%%%%%%%%%%%% my own implementation %%%%%%%%%%%%%%%%
%         disp(size(pixels,1));   % number of pixels
%         disp(size(pixels,2));   % 3: RGB
        % initialize k cluster centers {c1,c2,...,ck}, randomly
        % this should give us a K * 3 matrix
        % among size(pixels,1), select K rows as the initial centroids
        centroid = pixels(randsample(size(pixels,1), K), :);
%         disp("Initial centroids: ");
%         disp(centroid);
        previous_centroid = centroid + 10;
%         assignin("previous_centroid", previous_centroid);
        % initialize `class`
        class = zeros(size(pixels, 1), 1);
        % check if the cluster centers change
        tic
        while (norm(centroid - previous_centroid, 'fro') > 0)
%             disp("move on");
            previous_centroid = centroid;
            % cluster assignment: update class
            for i = 1: size(pixels,1)
                min_distance = Inf;
                nearest_j = 1;
                for j = 1: K
                    tmp_distance = norm(pixels(i,:) - centroid(j,:))^2;
                    if (tmp_distance < min_distance)
                        min_distance = tmp_distance;
                        nearest_j = j;
                    end
                end
                % update `class`
                class(i,:) = nearest_j;
            end
            % center adjustment
            for j = 1: K
                isAssigned = (class.' == j);
                num_points = nnz(isAssigned);
%                 disp(num_points);
%                 disp("===")
                if(num_points == 0)
                    centroid(j,:) = NaN(1,size(pixels,2));
                else
                    centroid(j,:) = isAssigned * pixels ./ num_points;
                end
            end
        end
        disp("K-means runtime: ")
        disp(toc);
        
%         disp("Number of filled clusters: ");
%         cluster_numPoints = zeros(1, K);
%         for row=1 : size(class, 1)
%             cluster_numPoints(class(row,:)) = cluster_numPoints(class(row,:)) + 1;
%         end
%         disp(sum(cluster_numPoints.' ~= 0));
          %%%%%%%%%% for emtpy cluster, centroid set to be NaN, NaN, NaN, so
          %%%%%%%%%% it's not a problem

end
