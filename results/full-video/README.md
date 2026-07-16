# Full-video benchmark artifacts

These files were produced by the server-authoritative NRT run in Slurm job
`<job-id>` from
`results/full_video_final`.

- `dit_only_results.json` is the combined, validated result; the three
  `*_result.json` files retain per-run measurements for eager,
  `torch.compile(max-autotune)`, and CODA.
- `artifacts/videos/` contains the three complete generated videos;
  `artifacts/gifs/` and `artifacts/aligned_frames/` contain the compact eager
  and CODA comparisons.
- The latency label is **DiT-only, T5/VAE excluded**. CUDA events cover all 100
  DiT forwards in each 50-step CFG generation; T5, scheduler work, VAE decode,
  and export are outside the latency metric.
- Raw VAE-decoded RGB arrays were intentionally retained on NRT for similarity
  analysis and are not included in Git. Reported PSNR/SSIM are computed from
  those arrays before MP4 encoding.
- `SHA256SUMS` covers every included JSON and media artifact plus this README.
