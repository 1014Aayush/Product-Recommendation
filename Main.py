import pandas as pd
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges, negative_sampling
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import BCEWithLogitsLoss, Dropout, BatchNorm1d
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

print("=== LOADING AND ANALYZING DATA ===")

# Load Data
ratings = pd.read_csv("./ml-100k/u.data", sep='\t', names=["user_id", "item_id", "rating", "timestamp"])
movies = pd.read_csv(
    "./ml-100k/u.item", sep='|', encoding='latin-1', header=None,
    names=['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action',
           'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
           'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
)

print(f"Original dataset: {len(ratings)} ratings, {ratings['user_id'].nunique()} users, {ratings['item_id'].nunique()} items")
print(f"Rating distribution:")
print(ratings['rating'].value_counts().sort_index())

# Plot rating distribution
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
ratings['rating'].hist(bins=5, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
user_rating_counts = ratings.groupby('user_id').size()
user_rating_counts.hist(bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
plt.title('User Activity Distribution')
plt.xlabel('Number of Ratings per User')
plt.ylabel('Number of Users')
plt.tight_layout()
plt.savefig('rating_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Filter ratings >= 4 for implicit feedback
ratings_filtered = ratings[ratings['rating'] >= 4].copy()
print(f"\nFiltered dataset (rating >= 4): {len(ratings_filtered)} interactions")
print(f"Users: {ratings_filtered['user_id'].nunique()}, Items: {ratings_filtered['item_id'].nunique()}")

# Genre analysis
genre_cols = movies.columns[6:25]
genre_popularity = movies[genre_cols].sum().sort_values(ascending=False)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
genre_popularity.plot(kind='bar', color='orange', alpha=0.8)
plt.title('Genre Popularity')
plt.xlabel('Genres')
plt.ylabel('Number of Movies')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
movie_genre_count = movies[genre_cols].sum(axis=1)
movie_genre_count.hist(bins=20, alpha=0.7, color='purple', edgecolor='black')
plt.title('Number of Genres per Movie')
plt.xlabel('Genre Count')
plt.ylabel('Number of Movies')
plt.tight_layout()
plt.savefig('genre_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("=== BUILDING GRAPH STRUCTURE ===")

# Build homogeneous graph (user-user and item-item connections)
user_item_matrix = ratings_filtered.pivot_table(index='user_id', columns='item_id', values='rating', fill_value=0)
print(f"User-item matrix shape: {user_item_matrix.shape}")

# Calculate similarities
print("Computing user similarities...")
user_similarity = np.corrcoef(user_item_matrix.values)
print("Computing item similarities...")
item_similarity = np.corrcoef(user_item_matrix.values.T)

# Replace NaN with 0
user_similarity = np.nan_to_num(user_similarity, 0)
item_similarity = np.nan_to_num(item_similarity, 0)

# Analyze similarity distributions
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
user_sim_flat = user_similarity[np.triu_indices_from(user_similarity, k=1)]
plt.hist(user_sim_flat, bins=50, alpha=0.7, color='red', edgecolor='black')
plt.title('User Similarity Distribution')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
item_sim_flat = item_similarity[np.triu_indices_from(item_similarity, k=1)]
plt.hist(item_sim_flat, bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.title('Item Similarity Distribution')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
sparsity = 1.0 - (np.count_nonzero(user_item_matrix) / float(user_item_matrix.size))
plt.bar(['Matrix Sparsity'], [sparsity], color='green', alpha=0.7)
plt.title(f'Data Sparsity: {sparsity:.3f}')
plt.ylabel('Sparsity Ratio')
plt.tight_layout()
plt.savefig('similarity_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Threshold for connections
user_threshold = 0.3
item_threshold = 0.2

print(f"User similarity threshold: {user_threshold}")
print(f"Item similarity threshold: {item_threshold}")
print(f"User connections above threshold: {np.sum(user_similarity > user_threshold) - len(user_similarity)}")
print(f"Item connections above threshold: {np.sum(item_similarity > item_threshold) - len(item_similarity)}")

# Create node mappings
all_users = sorted(ratings_filtered['user_id'].unique())
all_items = sorted(ratings_filtered['item_id'].unique())
node_mapping = {}
reverse_mapping = {}

print(f"Total nodes: {len(all_users)} users + {len(all_items)} items = {len(all_users) + len(all_items)}")

# Users: 0 to len(users)-1
for i, user in enumerate(all_users):
    node_mapping[f'user_{user}'] = i
    reverse_mapping[i] = f'user_{user}'

# Items: len(users) to len(users)+len(items)-1
for i, item in enumerate(all_items):
    node_mapping[f'item_{item}'] = len(all_users) + i
    reverse_mapping[len(all_users) + i] = f'item_{item}'

# Build edge list
edges = []
edge_types = {'user_item': 0, 'user_user': 0, 'item_item': 0}

print("Building edges...")

# User-item edges (bipartite)
for _, row in ratings_filtered.iterrows():
    user_node = node_mapping[f'user_{row.user_id}']
    item_node = node_mapping[f'item_{row.item_id}']
    edges.append([user_node, item_node])
    edges.append([item_node, user_node])  # Undirected
    edge_types['user_item'] += 1

# User-user edges based on similarity
for i in range(len(all_users)):
    for j in range(i+1, len(all_users)):
        if user_similarity[i, j] > user_threshold:
            edges.append([i, j])
            edges.append([j, i])
            edge_types['user_user'] += 1

# Item-item edges based on similarity
for i in range(len(all_items)):
    for j in range(i+1, len(all_items)):
        if item_similarity[i, j] > item_threshold:
            item_i = len(all_users) + i
            item_j = len(all_users) + j
            edges.append([item_i, item_j])
            edges.append([item_j, item_i])
            edge_types['item_item'] += 1

print(f"Edge statistics:")
print(f"  User-Item edges: {edge_types['user_item']}")
print(f"  User-User edges: {edge_types['user_user']}")
print(f"  Item-Item edges: {edge_types['item_item']}")
print(f"  Total edges: {len(edges)}")

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# Plot graph statistics
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
edge_type_counts = [edge_types['user_item'], edge_types['user_user'], edge_types['item_item']]
plt.pie(edge_type_counts, labels=['User-Item', 'User-User', 'Item-Item'], autopct='%1.1f%%', startangle=90)
plt.title('Edge Type Distribution')

plt.subplot(1, 2, 2)
degrees = torch.bincount(edge_index.flatten()).numpy()
plt.hist(degrees, bins=50, alpha=0.7, color='teal', edgecolor='black')
plt.title('Node Degree Distribution')
plt.xlabel('Degree')
plt.ylabel('Number of Nodes')
plt.yscale('log')
plt.tight_layout()
plt.savefig('graph_statistics.png', dpi=300, bbox_inches='tight')
plt.show()

print("=== CREATING NODE FEATURES ===")

# Enhanced node features
def create_node_features():
    features = []
    
    # User features
    user_stats = ratings_filtered.groupby('user_id').agg({
        'rating': ['mean', 'count', 'std'],
        'item_id': 'nunique'
    }).reset_index()
    user_stats.columns = ['user_id', 'avg_rating', 'num_ratings', 'rating_std', 'num_unique_items']
    user_stats['rating_std'] = user_stats['rating_std'].fillna(0)
    
    print(f"User statistics:")
    print(f"  Average ratings per user: {user_stats['num_ratings'].mean():.2f}")
    print(f"  Max ratings per user: {user_stats['num_ratings'].max()}")
    print(f"  Min ratings per user: {user_stats['num_ratings'].min()}")
    
    # Genre preferences for users
    print("Computing user genre preferences...")
    user_genre_prefs = []
    for user in all_users:
        user_items = ratings_filtered[ratings_filtered['user_id'] == user]['item_id'].values
        user_movies = movies[movies['movie_id'].isin(user_items)]
        genre_cols = movies.columns[6:25]
        genre_pref = user_movies[genre_cols].mean().values
        genre_pref = np.nan_to_num(genre_pref, 0)
        user_genre_prefs.append(genre_pref)
    
    user_genre_prefs = np.array(user_genre_prefs)
    
    for i, user in enumerate(all_users):
        user_stat = user_stats[user_stats['user_id'] == user].iloc[0]
        feat = [
            1, 0,  # user indicator
            user_stat['avg_rating'] / 5.0,
            min(user_stat['num_ratings'] / 100.0, 1.0),
            user_stat['rating_std'] / 2.0,
            min(user_stat['num_unique_items'] / 50.0, 1.0)
        ]
        feat.extend(user_genre_prefs[i])
        features.append(feat)
    
    # Item features
    item_stats = ratings_filtered.groupby('item_id').agg({
        'rating': ['mean', 'count', 'std']
    }).reset_index()
    item_stats.columns = ['item_id', 'avg_rating', 'num_ratings', 'rating_std']
    item_stats['rating_std'] = item_stats['rating_std'].fillna(0)
    
    print(f"Item statistics:")
    print(f"  Average ratings per item: {item_stats['num_ratings'].mean():.2f}")
    print(f"  Max ratings per item: {item_stats['num_ratings'].max()}")
    print(f"  Min ratings per item: {item_stats['num_ratings'].min()}")
    
    for item in all_items:
        if item in item_stats['item_id'].values:
            item_stat = item_stats[item_stats['item_id'] == item].iloc[0]
            avg_rating = item_stat['avg_rating'] / 5.0
            popularity = min(item_stat['num_ratings'] / 100.0, 1.0)
            rating_std = item_stat['rating_std'] / 2.0
        else:
            avg_rating = 0.6
            popularity = 0.1
            rating_std = 0.2
            
        movie_info = movies[movies['movie_id'] == item]
        if len(movie_info) > 0:
            genres = movie_info.iloc[0, 6:25].values.astype(float)
        else:
            genres = np.zeros(19)
            
        feat = [0, 1, avg_rating, popularity, rating_std, 0]  # item indicator
        feat.extend(genres)
        features.append(feat)
    
    return np.array(features, dtype=np.float32)

node_features = create_node_features()
print(f"Node features shape: {node_features.shape}")

# Visualize feature distributions
plt.figure(figsize=(15, 10))
feature_names = ['is_user', 'is_item', 'avg_rating', 'popularity', 'rating_std', 'diversity'] + list(genre_cols)

for i in range(min(12, node_features.shape[1])):
    plt.subplot(3, 4, i+1)
    plt.hist(node_features[:, i], bins=30, alpha=0.7, edgecolor='black')
    plt.title(f'{feature_names[i] if i < len(feature_names) else f"Feature {i}"}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# Normalize features
print("Normalizing features...")
scaler = StandardScaler()
node_features_normalized = scaler.fit_transform(node_features)
x = torch.tensor(node_features_normalized, dtype=torch.float)

print(f"Feature statistics after normalization:")
print(f"  Mean: {node_features_normalized.mean():.4f}")
print(f"  Std: {node_features_normalized.std():.4f}")

# Create positive edges for training (user-item interactions)
pos_edges = []
for _, row in ratings_filtered.iterrows():
    user_node = node_mapping[f'user_{row.user_id}']
    item_node = node_mapping[f'item_{row.item_id}']
    pos_edges.append([user_node, item_node])

pos_edge_index = torch.tensor(pos_edges, dtype=torch.long).t().contiguous()
print(f"Positive edges for training: {pos_edge_index.shape[1]}")

# Create PyG data
data = Data(x=x, edge_index=edge_index, pos_edge_index=pos_edge_index)

print("=== MODEL ARCHITECTURE ===")

# Improved model architecture
class ImprovedGNN(nn.Module):
    def __init__(self, num_features, hidden_dim=64, num_layers=3, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(num_features, hidden_dim))
        self.bns.append(BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(BatchNorm1d(hidden_dim))
            
        # Output layer
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.bns.append(BatchNorm1d(hidden_dim))
        
        # Link prediction head
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x, edge_index):
        # Graph convolution
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def predict_link(self, x, edge_index):
        embeddings = self.forward(x, edge_index)
        src, dst = edge_index
        edge_emb = torch.cat([embeddings[src], embeddings[dst]], dim=1)
        return self.link_predictor(edge_emb).squeeze()

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = ImprovedGNN(num_features=x.shape[1], hidden_dim=128, num_layers=3, dropout=0.2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Model architecture:")
print(model)

data = data.to(device)

print("=== TRAINING PREPARATION ===")

# Split edges for training/validation/test
def split_edges(pos_edge_index, num_nodes, val_ratio=0.1, test_ratio=0.1):
    num_edges = pos_edge_index.size(1)
    perm = torch.randperm(num_edges)
    
    num_val = int(val_ratio * num_edges)
    num_test = int(test_ratio * num_edges)
    num_train = num_edges - num_val - num_test
    
    train_edges = pos_edge_index[:, perm[:num_train]]
    val_edges = pos_edge_index[:, perm[num_train:num_train + num_val]]
    test_edges = pos_edge_index[:, perm[num_train + num_val:]]
    
    return train_edges, val_edges, test_edges

train_pos, val_pos, test_pos = split_edges(data.pos_edge_index, x.size(0))

print(f"Data splits:")
print(f"  Training edges: {train_pos.shape[1]}")
print(f"  Validation edges: {val_pos.shape[1]}")
print(f"  Test edges: {test_pos.shape[1]}")

# Training tracking
train_losses = []
val_aucs = []
test_aucs = []
learning_rates = []

def train_epoch():
    model.train()
    optimizer.zero_grad()
    
    # Positive samples
    pos_pred = model.predict_link(data.x, train_pos)
    
    # Negative sampling
    neg_edge_index = negative_sampling(
        train_pos, num_nodes=data.x.size(0), 
        num_neg_samples=train_pos.size(1),
        method='sparse'
    )
    neg_pred = model.predict_link(data.x, neg_edge_index)
    
    # Binary classification loss
    pos_loss = F.binary_cross_entropy_with_logits(pos_pred, torch.ones_like(pos_pred))
    neg_loss = F.binary_cross_entropy_with_logits(neg_pred, torch.zeros_like(neg_pred))
    loss = pos_loss + neg_loss
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item(), pos_loss.item(), neg_loss.item()

@torch.no_grad()
def evaluate(pos_edges, return_predictions=False):
    model.eval()
    
    pos_pred = torch.sigmoid(model.predict_link(data.x, pos_edges))
    
    neg_edge_index = negative_sampling(
        pos_edges, num_nodes=data.x.size(0),
        num_neg_samples=pos_edges.size(1),
        method='sparse'
    )
    neg_pred = torch.sigmoid(model.predict_link(data.x, neg_edge_index))
    
    pred = torch.cat([pos_pred, neg_pred]).cpu().numpy()
    label = np.concatenate([np.ones(pos_pred.size(0)), np.zeros(neg_pred.size(0))])
    
    auc = roc_auc_score(label, pred)
    
    if return_predictions:
        return auc, pred, label
    return auc

print("=== STARTING TRAINING ===")

# Training loop
best_val_auc = 0
patience_counter = 0
patience = 20

for epoch in range(1, 201):
    loss, pos_loss, neg_loss = train_epoch()
    train_losses.append(loss)
    
    if epoch % 10 == 0:
        val_auc = evaluate(val_pos)
        test_auc = evaluate(test_pos)
        current_lr = optimizer.param_groups[0]['lr']
        
        val_aucs.append(val_auc)
        test_aucs.append(test_auc)
        learning_rates.append(current_lr)
        
        print(f'Epoch {epoch:03d} | Loss: {loss:.4f} (Pos: {pos_loss:.4f}, Neg: {neg_loss:.4f}) | Val AUC: {val_auc:.4f} | Test AUC: {test_auc:.4f} | LR: {current_lr:.6f}')
        
        scheduler.step(loss)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            print(f"  ✓ New best validation AUC!")
        else:
            patience_counter += 10
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

print("=== TRAINING COMPLETE ===")

# Plot training curves
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Training Loss', color='red', alpha=0.7)
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
epochs_recorded = range(10, len(val_aucs)*10 + 1, 10)
plt.plot(epochs_recorded, val_aucs, label='Validation AUC', color='blue', marker='o')
plt.plot(epochs_recorded, test_aucs, label='Test AUC', color='green', marker='s')
plt.title('Model Performance Over Time')
plt.xlabel('Epoch')
plt.ylabel('AUC Score')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(epochs_recorded, learning_rates, label='Learning Rate', color='purple', marker='^')
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print("=== FINAL EVALUATION ===")

# Final evaluation with detailed metrics
final_test_auc, test_pred, test_label = evaluate(test_pos, return_predictions=True)
print(f'Final Test AUC: {final_test_auc:.4f}')

# ROC and Precision-Recall curves
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score

fpr, tpr, _ = roc_curve(test_label, test_pred)
precision, recall, _ = precision_recall_curve(test_label, test_pred)
ap_score = average_precision_score(test_label, test_pred)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {final_test_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {ap_score:.4f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.hist(test_pred[test_label == 1], bins=50, alpha=0.7, label='Positive', color='green', density=True)
plt.hist(test_pred[test_label == 0], bins=50, alpha=0.7, label='Negative', color='red', density=True)
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.title('Prediction Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('evaluation_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# Confusion matrix at different thresholds
thresholds = [0.3, 0.5, 0.7]
plt.figure(figsize=(15, 5))

for i, threshold in enumerate(thresholds):
    binary_pred = (test_pred > threshold).astype(int)
    cm = confusion_matrix(test_label, binary_pred)
    
    plt.subplot(1, 3, i+1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True)
    plt.title(f'Confusion Matrix (threshold={threshold})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()

print("=== EMBEDDING ANALYSIS ===")

# Get final embeddings
with torch.no_grad():
    model.eval()
    embeddings = model(data.x, data.edge_index).cpu().numpy()

print(f"Embedding shape: {embeddings.shape}")

# Separate user and item embeddings
user_embeddings = embeddings[:len(all_users)]
item_embeddings = embeddings[len(all_users):]

print(f"User embeddings: {user_embeddings.shape}")
print(f"Item embeddings: {item_embeddings.shape}")

# PCA visualization
pca = PCA(n_components=2)
user_pca = pca.fit_transform(user_embeddings)
item_pca = pca.transform(item_embeddings)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(user_pca[:, 0], user_pca[:, 1], alpha=0.6, s=20, color='blue', label='Users')
plt.scatter(item_pca[:, 0], item_pca[:, 1], alpha=0.6, s=20, color='red', label='Items')
plt.title('PCA Visualization of Embeddings')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.legend()
plt.grid(True, alpha=0.3)

# t-SNE visualization (sample for performance)
if len(embeddings) > 1000:
    sample_indices = np.random.choice(len(embeddings), 1000, replace=False)
    sample_embeddings = embeddings[sample_indices]
    sample_labels = ['User' if i < len(all_users) else 'Item' for i in sample_indices]
else:
    sample_embeddings = embeddings
    sample_labels = ['User'] * len(all_users) + ['Item'] * len(all_items)

print("Computing t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_tsne = tsne.fit_transform(sample_embeddings)

plt.subplot(1, 3, 2)
user_mask = np.array(sample_labels) == 'User'
plt.scatter(embeddings_tsne[user_mask, 0], embeddings_tsne[user_mask, 1], 
           alpha=0.6, s=20, color='blue', label='Users')
plt.scatter(embeddings_tsne[~user_mask, 0], embeddings_tsne[~user_mask, 1], 
           alpha=0.6, s=20, color='red', label='Items')
plt.title('t-SNE Visualization of Embeddings')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend()
plt.grid(True, alpha=0.3)

# Embedding similarity heatmap (sample)
sample_size = min(50, len(all_users), len(all_items))
user_sample = user_embeddings[:sample_size]
item_sample = item_embeddings[:sample_size]

similarity_matrix = np.dot(user_sample, item_sample.T)
plt.subplot(1, 3, 3)
sns.heatmap(similarity_matrix, cmap='coolwarm', center=0, square=True)
plt.title(f'User-Item Similarity Matrix\n(Sample {sample_size}x{sample_size})')
plt.xlabel('Items')
plt.ylabel('Users')

plt.tight_layout()
plt.savefig('embedding_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Embedding statistics
print(f"Embedding statistics:")
print(f"  User embedding mean: {user_embeddings.mean():.4f}")
print(f"  User embedding std: {user_embeddings.std():.4f}")
print(f"  Item embedding mean: {item_embeddings.mean():.4f}")
print(f"  Item embedding std: {item_embeddings.std():.4f}")

# Cosine similarity analysis
from sklearn.metrics.pairwise import cosine_similarity

user_cosine_sim = cosine_similarity(user_embeddings)
item_cosine_sim = cosine_similarity(item_embeddings)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
user_sim_values = user_cosine_sim[np.triu_indices_from(user_cosine_sim, k=1)]
plt.hist(user_sim_values, bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.title('User Embedding Cosine Similarity')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.axvline(user_sim_values.mean(), color='red', linestyle='--', label=f'Mean: {user_sim_values.mean():.3f}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
item_sim_values = item_cosine_sim[np.triu_indices_from(item_cosine_sim, k=1)]
plt.hist(item_sim_values, bins=50, alpha=0.7, color='red', edgecolor='black')
plt.title('Item Embedding Cosine Similarity')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.axvline(item_sim_values.mean(), color='blue', linestyle='--', label=f'Mean: {item_sim_values.mean():.3f}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
# Cross similarity (user-item)
cross_sim = cosine_similarity(user_embeddings[:100], item_embeddings[:100])
cross_sim_values = cross_sim.flatten()
plt.hist(cross_sim_values, bins=50, alpha=0.7, color='green', edgecolor='black')
plt.title('User-Item Embedding Cosine Similarity')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.axvline(cross_sim_values.mean(), color='orange', linestyle='--', label=f'Mean: {cross_sim_values.mean():.3f}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('similarity_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

print("=== RECOMMENDATION ANALYSIS ===")

# Recommendation function
@torch.no_grad()
def recommend_for_user(user_id, top_k=10, return_scores=False):
    model.eval()
    embeddings = model(data.x, data.edge_index)
    
    if f'user_{user_id}' not in node_mapping:
        return []
    
    user_idx = node_mapping[f'user_{user_id}']
    user_emb = embeddings[user_idx]
    
    # Get all item embeddings
    item_indices = [node_mapping[f'item_{item}'] for item in all_items]
    item_embs = embeddings[item_indices]
    
    # Compute scores
    scores = torch.cosine_similarity(user_emb.unsqueeze(0), item_embs, dim=1)
    
    # Get user's rated items
    rated_items = set(ratings_filtered[ratings_filtered['user_id'] == user_id]['item_id'].values)
    
    # Get top recommendations
    _, top_indices = torch.topk(scores, min(len(all_items), top_k * 2))
    
    recommendations = []
    rec_scores = []
    for idx in top_indices:
        item_id = all_items[idx]
        if item_id not in rated_items:
            movie_title = movies[movies['movie_id'] == item_id]['title'].iloc[0] if len(movies[movies['movie_id'] == item_id]) > 0 else f"Movie {item_id}"
            recommendations.append(movie_title)
            rec_scores.append(scores[idx].item())
        if len(recommendations) >= top_k:
            break
    
    if return_scores:
        return recommendations, rec_scores
    return recommendations

# Test recommendations for multiple users
test_users = [5, 10, 50, 100, 200]
all_recommendations = {}

print("Generating recommendations for test users...")
for user_id in test_users:
    if user_id in all_users:
        recs, scores = recommend_for_user(user_id, top_k=10, return_scores=True)
        all_recommendations[user_id] = (recs, scores)
        
        print(f"\nUser {user_id} - Top 10 recommendations:")
        user_ratings = ratings[ratings['user_id'] == user_id]
        print(f"  User has rated {len(user_ratings)} movies with avg rating {user_ratings['rating'].mean():.2f}")
        
        for i, (title, score) in enumerate(zip(recs, scores), 1):
            print(f"  {i:2d}. {title:<50} (Score: {score:.3f})")

# Recommendation score distribution
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
all_scores = []
for user_id, (recs, scores) in all_recommendations.items():
    all_scores.extend(scores)

plt.hist(all_scores, bins=30, alpha=0.7, color='purple', edgecolor='black')
plt.title('Recommendation Score Distribution')
plt.xlabel('Cosine Similarity Score')
plt.ylabel('Frequency')
plt.axvline(np.mean(all_scores), color='red', linestyle='--', label=f'Mean: {np.mean(all_scores):.3f}')
plt.legend()
plt.grid(True, alpha=0.3)

# Genre diversity in recommendations
plt.subplot(1, 3, 2)
genre_counts = np.zeros(len(genre_cols))
total_recs = 0

for user_id, (recs, _) in all_recommendations.items():
    for rec in recs:
        movie_info = movies[movies['title'] == rec]
        if len(movie_info) > 0:
            genres = movie_info.iloc[0, 6:25].values
            genre_counts += genres
            total_recs += 1

genre_percentages = genre_counts / total_recs * 100
plt.bar(range(len(genre_cols)), genre_percentages, color='orange', alpha=0.8)
plt.title('Genre Distribution in Recommendations')
plt.xlabel('Genres')
plt.ylabel('Percentage of Recommendations')
plt.xticks(range(len(genre_cols)), genre_cols, rotation=45)

# Coverage analysis
plt.subplot(1, 3, 3)
all_recommended_items = set()
for user_id, (recs, _) in all_recommendations.items():
    for rec in recs:
        movie_info = movies[movies['title'] == rec]
        if len(movie_info) > 0:
            all_recommended_items.add(movie_info.iloc[0]['movie_id'])

coverage = len(all_recommended_items) / len(all_items) * 100
popularity_bias = []

for user_id, (recs, _) in all_recommendations.items():
    user_pop_bias = 0
    for rec in recs:
        movie_info = movies[movies['title'] == rec]
        if len(movie_info) > 0:
            movie_id = movie_info.iloc[0]['movie_id']
            popularity = len(ratings[ratings['item_id'] == movie_id])
            user_pop_bias += popularity
    popularity_bias.append(user_pop_bias / len(recs))

plt.bar(['Coverage', 'Avg Popularity'], [coverage, np.mean(popularity_bias)], 
        color=['green', 'red'], alpha=0.7)
plt.title('Recommendation Quality Metrics')
plt.ylabel('Value')
plt.text(0, coverage/2, f'{coverage:.1f}%', ha='center', va='center', fontweight='bold')
plt.text(1, np.mean(popularity_bias)/2, f'{np.mean(popularity_bias):.1f}', ha='center', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('recommendation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nRecommendation System Performance Summary:")
print(f"  Coverage: {coverage:.2f}% of items recommended")
print(f"  Average popularity bias: {np.mean(popularity_bias):.2f}")
print(f"  Average recommendation score: {np.mean(all_scores):.4f}")

# Cold start analysis
print("\n=== COLD START ANALYSIS ===")

# Find users with very few ratings
user_rating_counts = ratings_filtered.groupby('user_id').size()
cold_users = user_rating_counts[user_rating_counts <= 5].index.tolist()
warm_users = user_rating_counts[user_rating_counts > 20].index.tolist()

print(f"Cold users (≤5 ratings): {len(cold_users)}")
print(f"Warm users (>20 ratings): {len(warm_users)}")

# Compare recommendation quality
cold_sample = np.random.choice(cold_users, min(5, len(cold_users)), replace=False)
warm_sample = np.random.choice(warm_users, min(5, len(warm_users)), replace=False)

cold_scores = []
warm_scores = []

for user_id in cold_sample:
    if user_id in all_users:
        _, scores = recommend_for_user(user_id, top_k=10, return_scores=True)
        cold_scores.extend(scores)

for user_id in warm_sample:
    if user_id in all_users:
        _, scores = recommend_for_user(user_id, top_k=10, return_scores=True)
        warm_scores.extend(scores)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(cold_scores, bins=20, alpha=0.7, label='Cold Users', color='blue', density=True)
plt.hist(warm_scores, bins=20, alpha=0.7, label='Warm Users', color='red', density=True)
plt.title('Recommendation Quality: Cold vs Warm Users')
plt.xlabel('Recommendation Score')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
categories = ['Cold Users', 'Warm Users']
mean_scores = [np.mean(cold_scores) if cold_scores else 0, np.mean(warm_scores) if warm_scores else 0]
plt.bar(categories, mean_scores, color=['blue', 'red'], alpha=0.7)
plt.title('Average Recommendation Scores')
plt.ylabel('Average Score')
for i, score in enumerate(mean_scores):
    plt.text(i, score/2, f'{score:.3f}', ha='center', va='center', fontweight='bold', color='white')

plt.tight_layout()
plt.savefig('cold_start_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Cold users average score: {np.mean(cold_scores) if cold_scores else 0:.4f}")
print(f"Warm users average score: {np.mean(warm_scores) if warm_scores else 0:.4f}")

print("\n=== MODEL INSIGHTS SUMMARY ===")
print(f"✓ Dataset: {len(ratings)} total ratings, {len(ratings_filtered)} high-quality interactions")
print(f"✓ Graph: {len(edges)} edges ({edge_types['user_item']} user-item, {edge_types['user_user']} user-user, {edge_types['item_item']} item-item)")
print(f"✓ Features: {x.shape[1]} dimensions per node")
print(f"✓ Model: {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"✓ Performance: {final_test_auc:.4f} Test AUC, {ap_score:.4f} Average Precision")
print(f"✓ Recommendations: {coverage:.2f}% item coverage, {np.mean(all_scores):.4f} avg similarity")
print(f"✓ Training: {len(train_losses)} epochs, early stopping applied")

print("\nAll visualizations saved as PNG files:")
print("  - rating_analysis.png")
print("  - genre_analysis.png") 
print("  - similarity_analysis.png")
print("  - graph_statistics.png")
print("  - feature_distributions.png")
print("  - training_curves.png")
print("  - evaluation_metrics.png")
print("  - confusion_matrices.png")
print("  - embedding_analysis.png")
print("  - similarity_distributions.png")
print("  - recommendation_analysis.png")
print("  - cold_start_analysis.png")

print("\n" + "="*50)
print("TRAINING AND ANALYSIS COMPLETE!")
print("="*50)
