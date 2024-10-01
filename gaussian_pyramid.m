function out = gaussian_pyramid(img, level, h)
    if nargin < 3
        h = 1/16 * [1, 4, 6, 4, 1];  % 默认的高斯核
    end
    filt = h' * h;
    out{1} = imfilter(img, filt, 'replicate', 'conv');
    temp_img = img;
    for i = 2 : level
        temp_img = temp_img(1 : 2 : end, 1 : 2 : end);
        out{i} = imfilter(temp_img, filt, 'replicate', 'conv');
    end
end