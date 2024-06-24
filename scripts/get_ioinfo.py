import onnx

def get_model_io_info(model_path):
    # 加载模型
    model = onnx.load(model_path)
    graph = model.graph
    
    # 获取输入和输出信息
    input_info = {inp.name: [dim.dim_value for dim in inp.type.tensor_type.shape.dim] for inp in graph.input}
    output_info = {out.name: [dim.dim_value for dim in out.type.tensor_type.shape.dim] for out in graph.output}
    
    return input_info, output_info

def print_io_info(input_info, output_info):
    print("Model Inputs:")
    for name, shape in input_info.items():
        print(f"Name: {name}, Shape: {shape}")

    print("\nModel Outputs:")
    for name, shape in output_info.items():
        print(f"Name: {name}, Shape: {shape}")

if __name__ == "__main__":
    model_path = "../encoder-epoch-99-avg-1.onnx"  # 替换为你的模型路径
    input_info, output_info = get_model_io_info(model_path)
    print_io_info(input_info, output_info)
