function y = conv2_mult_full(a, B, convopt)
y = zeros((size(a,1)+size(B,1))-1,1,size(B,3));
for i=1:size(B,3)
    y(:,:,i) = conv2(a, B(:,:,i), convopt);
end