function [ class, centroid ] = mykmedoids( pixels, K )
%
% Your goal of this assignment is implementing your own K-medoids.
% Please refer to the instructions carefully, and we encourage you to
% consult with other resources about this algorithm on the web.
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
%        [class, centroid] = kmeans(pixels, K, "Distance","cityblock");
%         [class, centroid] = kmedoids(pixels, K, "Distance","cityblock");


        % initialize centroids
        % same as in the `mykmeans.m`
        centroid = pixels(randsample(size(pixels,1), K), :);
        previous_centroid = centroid + 10;
        class = zeros(size(pixels, 1), 1);
%         r = zeros(size(pixels, 1), K);    % not needed
        % check if it is Manhattan distance
%         disp(centroid(1,:) - previous_centroid(1,:));
%         disp(myDistance(centroid(1,:), previous_centroid(1,:)));
        
        % converge condition is independent to the distance difination
        tic
        while(norm(centroid - previous_centroid, 'fro') > 0)
            previous_centroid = centroid;
            for i = 1: size(pixels,1)
                min_distance = Inf;
                nearest_j = 1;
                for j = 1: K
                    tmp_distance = myDistance(pixels(i,:), centroid(j,:));
                    if (tmp_distance < min_distance)
                        min_distance = tmp_distance;
                        nearest_j = j;
                    end
                end
                % like in `mykmeans.m`, we actually don't need r^nk for we have
                % `class`
                class(i,:) = nearest_j;
               
            end
            % cluter adjustment
            % ref: https://www.geeksforgeeks.org/find-a-point-such-that-sum-of-the-manhattan-distances-is-minimized/
            % for Manhattan distance, we need middle elements of each of the 3 dimensions (RGB).
            % they would give us the minimun sum of Manhattan distance
            for j = 1: K
                isAssigned = (class.' == j).';
                % look into the data belonging to this cluster
                pixels_in_cluster = pixels(isAssigned,:);
                % "If A is a matrix, then sort(A) treats the columns of A as vectors and sorts each
                % column."
                sorted_pixels_in_cluster = sort(pixels_in_cluster);
                for baseColorIndex=1: size(pixels, 2)
                    if(isempty(sorted_pixels_in_cluster))
                        centroid(j,:) = NaN(1,size(pixels,2));
                    else
                        middle_idx = fix((size(sorted_pixels_in_cluster,1)+1) / 2);
                        centroid(j,:) = sorted_pixels_in_cluster(middle_idx,:);                        
                    end

                end
            end
        end
        disp("K-medoids runtime: ")
        disp(toc)
        
end

% Manhattan distance
function dis = myDistance(pointOne, pointTwo)
    dis = norm(pointOne - pointTwo, 1);
%     dis = norm(pointOne- pointTwo, "inf");
    
end