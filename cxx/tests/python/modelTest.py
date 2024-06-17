import ortcxx

def main():
    print("Load this model '../../../models/test_wb.onnx'")
    env = ortcxx.Model("../../../models/test_wb.onnx", True, 1, 3, 1)
    print("Load done")
    print("Load this model '../../models/test_wb.onnx'")
    env = ortcxx.Model("../../models/test_wb.onnx", True, 1, 3, 1) # This will raise an error
    print("Load done")
    
if __name__ == "__main__":
    main()