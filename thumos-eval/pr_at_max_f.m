function [precision, recall, f1] = pr_at_max_f(...
        detfilename, gtpath, subset, threshold, single_confidence_hack)
    % Evaluates the precision and recall at the max F score.
    %
    % Args: First 4 as for TH14evalDet.m.
    %   single_confidence_hack (bool): Enable a 'hack' that doesn't take a max
    %       over F scores if the detections have only one confidence value (if
    %       this hack is enabled, we compute prec/rec using all the detections,
    %       and then computes F1 score). Default: false.
    %
    %       This is necessary because TH14EvalDet returns precisions/recalls at
    %       different "thresholds" even when multiple detections have the same
    %       confidence (or when all detections have the same confidence). This
    %       hack takes care of the specific case where detections have exactly
    %       one confidence threshold.
    %
    % Returns:
    %   precision (float)
    %   recall (float)
    %   f1 (float)

    if nargin < 5
        single_confidence_hack = false;
    end

    % precision_recalls is a (num_classes, 1) struct array.
    [precision_recalls, ~, ~] = TH14evalDet(...
        detfilename, gtpath, subset, threshold);

    num_classes = size(precision_recalls, 1);
    precisions = zeros(num_classes, 1);
    recalls = zeros(num_classes, 1);
    for i = 1:num_classes
        category = precision_recalls(i).class;

        [~,~,~,~,confidences]=textread(detfilename,'%s%f%f%d%f');
        max_f = 0;
        max_f_index = 0;

        % If the 'single confidence hack' is enabled, then compute F score
        % using all detections, instead of computing the max F score across
        % 'thresholds.' (See comment for single_confidence_hack parameter for
        % more information)
        if single_confidence_hack && numel(unique(confidences)) == 1
            precisions(i) = precision_recalls(i).prec(end);
            recalls(i) = precision_recalls(i).rec(end);
            max_f = compute_fs(precisions(i), recalls(i));
        else
            recs = precision_recalls(i).rec;
            precs = precision_recalls(i).prec;
            fs = compute_fs(precs, recs);
            [max_f, max_f_index] = max(fs);
            precisions(i) = precs(max_f_index);
            recalls(i) = recs(max_f_index);
        end

        fprintf('Precison:%1.3f, Recall:%1.3f and F:%1.3f for %s\n',...
                precisions(i), recalls(i), max_f, category)
    end
    precision = mean(precisions);
    recall = mean(recalls);
    f1 = 2 * (precision * recall) / (precision + recall);

    fprintf('Mean Precison:%1.3f, Mean Recall:%1.3f and F:%1.3f\n',...
            precision, recall, f1)
end

function fs = compute_fs(precisions, recalls)
    %
    % Args:
    %   precisions (n, 1)
    %   recalls (n, 1)
    %
    % Returns:
    %   fs (n, 1): F score for each value of precision, recall
    fs = 2 * (precisions .* recalls) ./ (precisions + recalls);
end
