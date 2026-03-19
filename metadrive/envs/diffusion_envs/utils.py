def show_map(env, save_path=None, resolution=(1024, 1024)):
    """打印地图 block/road 信息，并用 matplotlib 显示完整俯视图。

    Args:
        env:        已 reset() 的 MetaDrive env 实例
        save_path:  若非 None，则将图片保存到该路径（如 "map.png"）
        resolution: 输出图片的像素分辨率，默认 (1024, 1024)
    """
    from metadrive.utils.draw_top_down_map import draw_top_down_map
    import matplotlib.pyplot as plt

    m = env.current_map
    # ---- 绘制并俯视图 ----
    img = draw_top_down_map(m, resolution=resolution, semantic_map=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=100)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"Top-down Map  (blocks={m.num_blocks})", fontsize=12)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[show_map] Map image saved to: {save_path}")
    plt.show()