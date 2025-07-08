import torch


class GaussKernel:
    def __init__(self, kernel_size=15, sigma=1.5, device=None, dtype=torch.float32):
        """
        Create a Gaussian kernel for convolution.
        
        Args:
            kernel_size (int): Size of the square kernel (should be odd)
            sigma (float): Standard deviation of the Gaussian
            device (torch.device, optional): Device to place the kernel on
            dtype (torch.dtype, optional): Data type for the kernel (default: torch.float32)
        """
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.kernel = self._create_kernel()
        
    def _create_kernel(self):
        """Create a 2D Gaussian kernel"""
        # Ensure kernel size is odd
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
            
        # Create a coordinate grid
        coords = torch.arange(self.kernel_size, device=self.device, dtype=self.dtype)
        center = self.kernel_size // 2
        
        # Calculate distance from center for each coordinate
        x = coords.repeat(self.kernel_size, 1)
        y = x.transpose(0, 1)
        grid = torch.stack([x - center, y - center], dim=0)
        
        # Calculate 2D Gaussian
        dist_squared = grid[0] ** 2 + grid[1] ** 2
        kernel = torch.exp(-dist_squared / (2 * self.sigma ** 2))
        
        # Normalize kernel so sum = 1
        kernel = kernel / kernel.sum()
        
        # Reshape for convolution [1, 1, kernel_size, kernel_size]
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        
        return kernel
    
    def get_kernel(self):
        """Return the Gaussian kernel"""
        return self.kernel
    
    def to(self, device):
        """Move kernel to specified device"""
        self.device = device
        self.kernel = self.kernel.to(device)
        return self
        
    def to_dtype(self, dtype):
        """Convert kernel to the specified data type"""
        self.dtype = dtype
        self.kernel = self.kernel.to(dtype=dtype)
        return self