function [precision, recall] = pr_at_max_recall(detfilename, gtpath, subset, threshold)
    % Evaluates the precision and recall at the lowest confidence threshold,
    % i.e. the maximum recall and precision at the maximum recall.
    %
    % Args: As for TH14evalDet.m.
    %
    % Returns:
    %   precision (float)
    %   recall (float)

    % precision_recalls is a (num_classes, 1) struct array.
    [precision_recalls, ~, ~] = TH14evalDet(...
        detfilename, gtpath, subset, threshold);

    num_classes = size(precision_recalls, 1);
    precisions = zeros(num_classes, 1);
    recalls = zeros(num_classes, 1);
    for i = 1:num_classes
        recalls(i) = precision_recalls(i).rec(end);
        precisions(i) = precision_recalls(i).prec(end);
    end
    precision = mean(precisions);
    recall = mean(recalls);
end
