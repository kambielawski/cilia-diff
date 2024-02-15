import taichi as ti

# Define whether to use the GPU or CPU
ti.init(arch=ti.gpu)  # Change to ti.cpu if GPUs are not available

# Create a field of one element
x = ti.field(dtype=ti.f32, shape=())

@ti.kernel
def add_to_x(val: ti.f32):
    x[None] += val

# Main Program
def main():
    print("Before addition:", x[None])

    # Add a value
    add_to_x(10.0)

    print("After addition:", x[None])

if __name__ == "__main__":
    main()