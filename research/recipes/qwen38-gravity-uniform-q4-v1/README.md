# q4 incumbent recipe metadata

This is the tensor-name -> segment-filename mapping from
, preserved here because it is
RECIPE metadata rather than weights.

The 13 GB of tensors under that directory hold no unique weight information: the clean
rebuild regenerated all 353 shared segments from the bf16 parent and every one compared
byte-identical, and the same check was repeated independently for variantB. The bytes
are a cache. This mapping is not, and without it the recipe cannot name which segment
file a tensor was packed into.

Preserving this file is the precondition for relegating those 13 GB.

sha256: 6b923c6315439439f942fe0cd5dcdf22f05823b18b2873f876c2397aa964a7e7
