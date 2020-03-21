cd '/Users/girinl/Downloads/2019-2020_RVAE/test_results_20200227'

name1 = 'man_440_';
k = 2;

cd WSJ0_simon_FFNN_VAE_latent_dim=16

name = [name1 'origin-' num2str(k) '.wav'];
[x0, fs] = audioread(name);

name = [name1 'recon-' num2str(k) '.wav'];
[x1, fs] = audioread(name);

L = length(x1);
x0_cut = x0(1:L);
a = x1\x0_cut;

err_1_raw = sqrt(sumsqr(x0_cut - x1)/L);
err_1_norm = sqrt(sumsqr(x0_cut - a*x1)/L);

cd ..
cd WSJ0_simon_RVAE_RNNenc_RNNdec_NoRecZ_latent_dim=16/

name = [name1 'origin-' num2str(k) '.wav'];
[x0, fs] = audioread(name);

name = [name1 'recon-' num2str(k) '.wav'];
[x2, fs] = audioread(name);

L = length(x2);
x0_cut = x0(1:L);
b = x2\x0_cut;

err_2_raw = sqrt(sumsqr(x0_cut - x2)/L);
err_2_norm = sqrt(sumsqr(x0_cut - b*x2)/L);


cd ..
cd WSJ0_simon_RVAE_RNNenc_RNNdec_RecZ_latent_dim=16/

name = [name1 'origin-' num2str(k) '.wav'];
[x0, fs] = audioread(name);

name = [name1 'recon-' num2str(k) '.wav'];
[x3, fs] = audioread(name);

L = length(x3);
x0_cut = x0(1:L);
c = x3\x0_cut;

err_3_raw = sqrt(sumsqr(x0_cut - x3)/L);
err_3_norm = sqrt(sumsqr(x0_cut - c*x3)/L);


cd ..
cd WSJ0_simon_RVAE_BRNNenc_BRNNdec_NoRecZ_latent_dim=16/

name = [name1 'origin-' num2str(k) '.wav'];
[x0, fs] = audioread(name);

name = [name1 'recon-' num2str(k) '.wav'];
[x4, fs] = audioread(name);

L = length(x4);
x0_cut = x0(1:L);
d = x4\x0_cut;

err_4_raw = sqrt(sumsqr(x0_cut - x4)/L);
err_4_norm = sqrt(sumsqr(x0_cut - d*x4)/L);


cd ..
cd WSJ0_simon_RVAE_BRNNenc_BRNNdec_RecZ_latent_dim=16/

name = [name1 'origin-' num2str(k) '.wav'];
[x0, fs] = audioread(name);

name = [name1 'recon-' num2str(k) '.wav'];
[x5, fs] = audioread(name);

L=length(x5);
x0_cut = x0(1:L);
e = x5\x0_cut;

err_5_raw = sqrt(sumsqr(x0_cut - x5)/L);
err_5_norm = sqrt(sumsqr(x0_cut - e*x5)/L);

err_raw = [err_1_raw err_2_raw err_3_raw err_4_raw err_5_raw]

err_norm = [err_1_norm err_2_norm err_3_norm err_4_norm err_5_norm]

figure(1); clf;
plot(x0,'g'); hold on; plot(x2/b,'b'); plot(x5/e,'r');
