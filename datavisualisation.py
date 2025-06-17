# c:\Users\atuly\Documents\GitHub\generatingsigma\datavisualisation.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_prices_data(filename='prices.txt'):
    """Load prices data from text file"""
    try:
        # Try to read as CSV first
        data = pd.read_csv(filename)
        return data
    except:
        # If CSV fails, try reading as space/tab separated
        try:
            data = pd.read_csv(filename, sep='\s+')
            return data
        except:
            # If structured reading fails, read as simple list
            with open(filename, 'r') as f:
                prices = []
                for line in f:
                    line = line.strip()
                    if line and line.replace('.', '').replace('-', '').isdigit():
                        prices.append(float(line))
            return pd.DataFrame({'price': prices})

def visualize_prices():
    """Create various visualizations of price data"""
    # Load data
    data = load_prices_data('prices.txt')
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Price Data Visualization', fontsize=16)
    
    # Plot 1: Line chart
    if 'price' in data.columns:
        axes[0, 0].plot(data.index, data['price'], marker='o', linewidth=2)
        axes[0, 0].set_title('Price Trend Over Time')
        axes[0, 0].set_xlabel('Index')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Histogram
    price_column = data.columns[0] if len(data.columns) > 0 else 'price'
    axes[0, 1].hist(data[price_column], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].set_title('Price Distribution')
    axes[0, 1].set_xlabel('Price')
    axes[0, 1].set_ylabel('Frequency')
    
    # Plot 3: Box plot
    axes[1, 0].boxplot(data[price_column], vert=True)
    axes[1, 0].set_title('Price Box Plot')
    axes[1, 0].set_ylabel('Price')
    
    # Plot 4: Scatter plot with trend
    axes[1, 1].scatter(data.index, data[price_column], alpha=0.6, color='red')
    z = np.polyfit(data.index, data[price_column], 1)
    p = np.poly1d(z)
    axes[1, 1].plot(data.index, p(data.index), "r--", alpha=0.8)
    axes[1, 1].set_title('Price Scatter with Trend Line')
    axes[1, 1].set_xlabel('Index')
    axes[1, 1].set_ylabel('Price')
    
    plt.tight_layout()
    plt.show()
    
    # Print basic statistics
    print("Price Statistics:")
    print(f"Count: {len(data)}")
    print(f"Mean: {data[price_column].mean():.2f}")
    print(f"Median: {data[price_column].median():.2f}")
    print(f"Min: {data[price_column].min():.2f}")
    print(f"Max: {data[price_column].max():.2f}")
    print(f"Standard Deviation: {data[price_column].std():.2f}")

if __name__ == "__main__":
    visualize_prices()