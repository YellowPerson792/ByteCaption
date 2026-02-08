"""
将Analysis目录和evaluation_samples下的所有SVG图像转换为PDF格式
并保存到experiment_results目录
"""

from pathlib import Path
import shutil
from typing import List

try:
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF
    HAS_SVGLIB = True
except ImportError:
    HAS_SVGLIB = False

try:
    import cairosvg
    HAS_CAIROSVG = True
except ImportError:
    HAS_CAIROSVG = False


def convert_svg_to_pdf_svglib(svg_path: Path, pdf_path: Path) -> bool:
    """使用svglib将SVG转换为PDF"""
    try:
        drawing = svg2rlg(str(svg_path))
        if drawing:
            renderPDF.drawToFile(drawing, str(pdf_path))
            return True
    except Exception as e:
        print(f"    svglib转换失败: {e}")
    return False


def convert_svg_to_pdf_cairo(svg_path: Path, pdf_path: Path) -> bool:
    """使用cairosvg将SVG转换为PDF"""
    try:
        cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
        return True
    except Exception as e:
        print(f"    cairosvg转换失败: {e}")
    return False


def convert_svg_to_pdf(svg_path: Path, pdf_path: Path) -> bool:
    """尝试多种方法转换SVG到PDF"""
    # 优先使用cairosvg（质量更好）
    if HAS_CAIROSVG:
        if convert_svg_to_pdf_cairo(svg_path, pdf_path):
            return True
    
    # 其次使用svglib
    if HAS_SVGLIB:
        if convert_svg_to_pdf_svglib(svg_path, pdf_path):
            return True
    
    print(f"    所有转换方法都失败了")
    return False


def collect_svg_files() -> List[Path]:
    """收集所有需要转换的SVG文件"""
    base_dir = Path(__file__).parent
    repo_root = base_dir.parent.parent
    
    svg_files = []
    
    # 1. Analysis目录下的SVG
    analysis_svgs = list(base_dir.glob("*.svg"))
    svg_files.extend(analysis_svgs)
    print(f"找到 {len(analysis_svgs)} 个Analysis SVG文件")
    
    # 2. evaluation_samples/bitstream_corruption_test目录下的SVG
    eval_dir = repo_root / "evaluation_samples" / "bitstream_corruption_test"
    if eval_dir.exists():
        eval_svgs = list(eval_dir.glob("*.svg"))
        svg_files.extend(eval_svgs)
        print(f"找到 {len(eval_svgs)} 个bitstream_corruption_test SVG文件")
    
    return svg_files


def main():
    """主函数"""
    # 检查依赖
    if not HAS_SVGLIB and not HAS_CAIROSVG:
        print("错误: 需要安装 cairosvg 或 svglib+reportlab")
        print("安装命令:")
        print("  pip install cairosvg")
        print("  或")
        print("  pip install svglib reportlab")
        return
    
    print(f"使用的转换库: {'cairosvg' if HAS_CAIROSVG else 'svglib+reportlab'}")
    
    # 创建输出目录
    output_dir = Path(__file__).parent / "experiment_results"
    output_dir.mkdir(exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 收集SVG文件
    print("\n收集SVG文件...")
    svg_files = collect_svg_files()
    
    if not svg_files:
        print("未找到SVG文件")
        return
    
    print(f"\n总共找到 {len(svg_files)} 个SVG文件")
    print("\n开始转换...")
    
    # 转换每个文件
    success_count = 0
    failed_files = []
    
    for svg_path in svg_files:
        pdf_name = svg_path.stem + ".pdf"
        pdf_path = output_dir / pdf_name
        
        print(f"\n转换: {svg_path.name} -> {pdf_name}")
        
        if convert_svg_to_pdf(svg_path, pdf_path):
            print(f"    ✓ 成功")
            success_count += 1
        else:
            print(f"    ✗ 失败")
            failed_files.append(svg_path.name)
    
    # 总结
    print("\n" + "="*60)
    print(f"转换完成!")
    print(f"成功: {success_count}/{len(svg_files)}")
    
    if failed_files:
        print(f"\n失败的文件:")
        for fname in failed_files:
            print(f"  - {fname}")
    
    print(f"\nPDF文件保存在: {output_dir}")


if __name__ == "__main__":
    main()
