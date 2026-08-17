from train_atlas_pose import ensure_image_cache, ensure_manifests


# Historical v6 cache helper; canonical v7 generates fixed-manifest batches through train_atlas_pose_v7.py.
if __name__ == "__main__":
    print([images.shape for images in ensure_image_cache(ensure_manifests())])
