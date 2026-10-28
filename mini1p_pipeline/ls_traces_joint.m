function T = ls_traces_joint(h5, S, F, mask, radius, hp_cutoff, use_gpu, block)
%LS_TRACES_JOINT  The least-squares step of extract_1p (least_squares_traces), as in rescue_missing_cells:
%   per-pixel mean removed, the same spatial high-pass and mask, T = (S'S)^-1 S'M, divided by the
%   footprint-weighted mean F. Every footprint in S is fitted jointly, so overlapping cells share their pixels.
%   T = ls_traces_joint(h5, S, F, mask, radius, hp_cutoff, use_gpu, block)   (T: frames x cells, h5 rows)
[H, W, C] = size(S);
hi = h5info(h5, '/mov');
n = hi.Dataspace.Size(3);
S2 = reshape(double(S), H * W, C);
Ssp = sparse(S2);
G = full(Ssp' * Ssp);
B = zeros(C, n);
for i0 = 1:block:n
    nb = min(block, n - i0 + 1);
    Mb = h5read(h5, '/mov', [1 1 i0], [H W nb]);
    Mb = bsxfun(@minus, single(Mb), single(F));
    if ~isinf(hp_cutoff)
        Mb = spatial_bandpass(Mb, radius, hp_cutoff, inf, use_gpu);
        if use_gpu, Mb = gather(Mb); end
    end
    if ~isempty(mask)
        Mb = bsxfun(@times, Mb, single(mask));
    end
    B(:, i0:i0 + nb - 1) = Ssp' * double(reshape(Mb, H * W, nb));
end
if rcond(G) > 1e-10
    X = G \ B;
else
    X = pinv(G) * B;
end
F_cell = (S2 ./ max(sum(S2, 1), eps))' * double(F(:));
T = single(bsxfun(@rdivide, X', F_cell'));
end
