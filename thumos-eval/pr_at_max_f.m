function [precision, recall, f1] = pr_at_max_f(...
        detfilename, gtpath, subset, threshold)
    % Evaluates the precision and recall at the max F score.
    %
    % Args: As for TH14evalDet.m.
    %
    % Returns:
    %   precision (float)
    %   recall (float)
    %   f1 (float)

    % precision_recalls is a (num_classes, 1) struct array.
    [precision_recalls, ~, ~] = TH14evalDet(...
        detfilename, gtpath, subset, threshold);

    num_classes = size(precision_recalls, 1);
    precisions = zeros(num_classes, 1);
    recalls = zeros(num_classes, 1);
    for i = 1:num_classes
        category = precision_recalls(i).class;

        recs = precision_recalls(i).rec;
        precs = precision_recalls(i).prec;
        fs = (2 * recs * precs) ./ (recs + precs);

        [~, max_index] = max(fs);
        precisions(i) = precs(max_index);
        recalls(i) = recs(max_index);
        fprintf('Precison:%1.3f, Recall:%1.3f and F:%1.3f for %s\n',...
                precisions(i), recalls(i), max_index, category)
    end
    precision = mean(precisions);
    recall = mean(recalls);
    f1 = 2 * (precision * recall) / (precision + recall);

    fprintf('Mean Precison:%1.3f, Mean Recall:%1.3f and F:%1.3f\n',...
            precision, recall, f1)
end
