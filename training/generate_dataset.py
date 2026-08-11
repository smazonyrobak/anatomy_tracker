from train_atlas_pose import ensure_image_cache, ensure_manifests


if __name__ == "__main__":
    print([images.shape for images in ensure_image_cache(ensure_manifests())])
