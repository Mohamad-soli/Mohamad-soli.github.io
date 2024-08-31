from pyDOE import lhs
import numpy as np
import matplotlib.pyplot as plt

def rotate(points, degree):
    """Rotate points around origin by degree."""
    theta = np.deg2rad(degree)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return np.dot(points, R)

def translate(points, dx, dy):
    """Translate points by dx and dy."""
    return points + np.array((dx, dy))

def scale(points, sx, sy):
    """Scale points by sx and sy."""
    return points * np.array((sx, sy))

def create_homogenous_coordinates(points):
    """Create homogenous coordinates for points."""
    return np.hstack((points, 1))

def calculate_point_distance(point1, point2):
    """Calculate distance between two points."""
    return np.sum((point1 - point2)**2, axis = 1)

def find_centroid(points):
    """Find centroid of points."""
    return np.mean(points, axis = 0)

def calculate_rotation(ref_point, closest_point, ref_centroid, src_centroid):
    """Calculate transformation between reference and source. 
       This algorithm is based on the paper "Least-Squares Fitting of Two 3-D Point Sets"
        K. S. ARUN, T. S. HUANG, AND S. D. BLOSTEIN - 1987
    """
    # Subtract the centroid from the point sets
    ref_point_prime = (ref_point - ref_centroid).reshape((2, 1))
    closest_point_prime = (closest_point - src_centroid).reshape((2, 1))
    # Step 2
    # Calculating the H matrix
    H = np.dot(ref_point_prime, closest_point_prime.T)
    # Step 3
    U, l, V = np.linalg.svd(H)
    # Step 4
    X = V.T @ U.T
    # Step 5
    # In this step we calculate the determinant of the matrix X to see if it can be a rotation matrix
    det_x = np.linalg.det(X)
    if det_x < 0:
        # X can not be a rotation matrix
        X = np.eye(2)
    return X

def calculate_translation(ref_centroid, src_centroid, R):
    """Calculate translation between reference and source."""
    return src_centroid - R @ ref_centroid
    

def icp(reference, source):
    """" This function creates a first estimate for the ICP algorithm"""
    # Calculate the centroid of the point sets
    ref_centroid = find_centroid(reference)
    # Find the centroid of the source set
    # Find the closest point to each set of pionts in the reference set from the source set
    R_history = np.zeros((2, 2))
    T_history = np.zeros((2, 1))
    ref_point_idx = 0
    loss = 1
    while loss>0.1:
        src_centroid = find_centroid(source)
        ref_point = reference[ref_point_idx]
        dists = calculate_point_distance(ref_point, source)
        closest_point = source[dists.argmin()]
        R = calculate_rotation(ref_point, closest_point, ref_centroid, src_centroid)
        T = calculate_translation(ref_centroid, src_centroid, R).reshape((2, 1))
        R_diff = np.linalg.norm(R - R_history)
        T_diff = np.linalg.norm(T - T_history)
        loss = R_diff + T_diff
        R_history = R
        T_history = T
        source_new = np.linalg.inv(R) @ source.T - T
        plot_transition(source, source_new.T)
        source = source_new.T
        plt.scatter(source[:, 0], source[:, 1], c = 'g', alpha=0.7)
        ref_point_idx += 1
    return R, T

def plot_transition(initial, final):
    """Plot the transition between initial and final."""
    dx_dy = final - initial
    for i in range(len(initial)):
        plt.arrow(initial[i, 0], initial[i, 1], dx_dy[i, 0], dx_dy[i, 1], color = 'r', alpha = 0.2, width=0.05)
        plt.title('Transition')
        plt.savefig(f'{figure_dir}transition.png')

if __name__ == '__main__':
    np.random.seed(123)
    figure_dir = './blog_posts/icp/figures/'
    # Generate a Latin Hypercube Sampling Design
    top_left_bound = np.array([5, 5])
    bottom_right_bound = np.array([0, 0])
    # Number of points to be generated
    N = 20
    # populating the page with points
    reference_points = bottom_right_bound + (top_left_bound-bottom_right_bound)*lhs(2, N)
    # plotting the points
    fig, ax = plt.subplots()
    # set axis equal
    ax.set_aspect('equal')
    ax.scatter(reference_points[:,0], reference_points[:,1], label = 'Base Points', s=80, facecolors='none', edgecolors='b', linewidths=3)

    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Base Points')
    ax.grid()
    plt.savefig(figure_dir + 'reference_points.png')
    
    # Second set of points by applying random rotation, translation and scaling to the base points
    # Rotation
    rotated_points = rotate(reference_points, 5)
    # Translation
    translated_points = translate(rotated_points, 2, 1)
    # Scaling
    source_points = scale(translated_points, 1, 1)
    # plotting the points
    # fig, ax = plt.subplots()
    ax.scatter(source_points[:,0], source_points[:,1], color='red', label = 'Transformed Points',  s=80, facecolors='none', edgecolors='r', linewidths=3)
    ax.legend()
    ax.set_title('Transformed Points')
    plt.savefig(figure_dir + 'source_points.png')
    # calculating the transformation matrix from line1 to line2
    R, T = icp(reference_points, source_points)

    plt.show()