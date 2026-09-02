/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

//
// BPlusTree.hpp — Standalone B+tree for ThermaState ColdLeaf.
//
// Root is always InternalNode* (uniform for simpler traversal/free).
// Internal nodes: up to ORDER keys, ORDER+1 children.
// Leaf nodes: up to ORDER key-value pairs, linked for range scans.
//
// Cost tracking: increments Hits::leaf_cost (comparison count) for training.
// Memory tracking: increments Hits::node_memory_count for allocation accounting.
//

#ifndef BPLUSTREE_H
#define BPLUSTREE_H

#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <utility>

namespace Hits {
    extern unsigned long long node_memory_count;
    extern long long leaf_cost;
    extern long long leaf_max_cost;
}

template<class key_T, class value_T>
class BPlusTree {
public:
    static constexpr int ORDER = 64;            // max keys/pairs per node
    static constexpr int MIN_KEYS = ORDER / 2;  // min for split/merge

    using slot_type = std::pair<key_T, value_T>;

    // ── node types ──────────────────────────────────────────

    struct LeafNode {
        int num_pairs;
        slot_type pairs[ORDER];
        LeafNode* next;  // range scan link

        LeafNode() : num_pairs(0), next(nullptr) {
            // zero-initialize pairs for clean valgrind
            std::memset(pairs, 0, sizeof(pairs));
        }
    };

    struct InternalNode {
        int num_keys;                        // 0..ORDER-1
        key_T keys[ORDER];                   // separators
        void* children[ORDER + 1];           // InternalNode* or LeafNode*
        bool leaf_children;                  // true if children are LeafNode*

        InternalNode() : num_keys(0), leaf_children(false) {
            std::memset(keys, 0, sizeof(keys));
            std::memset(children, 0, sizeof(children));
        }
    };

    // ── tree metadata ───────────────────────────────────────

    double lower;
    double upper;
    InternalNode* root;    // always InternalNode (never null after first insert/load)
    int size;              // total key-value pairs

    // ── memory helpers ──────────────────────────────────────

    static LeafNode* make_leaf() {
        auto* n = (LeafNode*) std::malloc(sizeof(LeafNode));
        new (n) LeafNode();
        Hits::node_memory_count += sizeof(LeafNode);
        return n;
    }

    static InternalNode* make_internal() {
        auto* n = (InternalNode*) std::malloc(sizeof(InternalNode));
        new (n) InternalNode();
        Hits::node_memory_count += sizeof(InternalNode);
        return n;
    }

    static void destroy_leaf(LeafNode* n) {
        Hits::node_memory_count -= sizeof(LeafNode);
        n->~LeafNode();
        std::free(n);
    }

    static void destroy_internal(InternalNode* n) {
        Hits::node_memory_count -= sizeof(InternalNode);
        n->~InternalNode();
        std::free(n);
    }

    static void destroy_tree(InternalNode* node) {
        if (!node) return;
        if (node->leaf_children) {
            for (int i = 0; i <= node->num_keys; ++i) {
                if (node->children[i]) destroy_leaf((LeafNode*) node->children[i]);
            }
        } else {
            for (int i = 0; i <= node->num_keys; ++i) {
                if (node->children[i]) destroy_tree((InternalNode*) node->children[i]);
            }
        }
        destroy_internal(node);
    }

    // ── factory ─────────────────────────────────────────────

    static BPlusTree* create(double lower, double upper) {
        auto* t = new BPlusTree();
        t->lower = lower;
        t->upper = upper;
        t->root = nullptr;
        t->size = 0;
        return t;
    }

    static void destroy(BPlusTree* tree) {
        if (!tree) return;
        if (tree->root) destroy_tree(tree->root);
        delete tree;
    }

    // ── leaf search (internal node traversal) ───────────────

    LeafNode* search_leaf(key_T key) const {
        InternalNode* cur = root;
        while (cur) {
            // binary search the separator keys
            // separator = max key of left child → route key <= sep to left
            int pos = cur->num_keys;
            int lo = 0, hi = cur->num_keys - 1;
            while (lo <= hi) {
                int mid = (lo + hi) >> 1;
                if (key <= cur->keys[mid]) {
                    pos = mid;
                    hi = mid - 1;
                } else {
                    lo = mid + 1;
                }
            }
            if (cur->leaf_children) {
                return (LeafNode*) cur->children[pos];
            }
            cur = (InternalNode*) cur->children[pos];
        }
        return nullptr;
    }

    LeafNode* search_leaf_cost(key_T key) const {
        InternalNode* cur = root;
        while (cur) {
            ++Hits::leaf_cost;
            Hits::leaf_max_cost = std::max<long long>(Hits::leaf_max_cost, 1);
            // separator = max key of left child → route key <= sep to left
            int pos = cur->num_keys;
            int lo = 0, hi = cur->num_keys - 1;
            while (lo <= hi) {
                int mid = (lo + hi) >> 1;
                ++Hits::leaf_cost;
                if (key <= cur->keys[mid]) {
                    pos = mid;
                    hi = mid - 1;
                } else {
                    lo = mid + 1;
                }
            }
            if (cur->leaf_children) {
                return (LeafNode*) cur->children[pos];
            }
            cur = (InternalNode*) cur->children[pos];
        }
        return nullptr;
    }

    // ── point query ─────────────────────────────────────────

    int find(key_T key) const {
        LeafNode* leaf = search_leaf(key);
        if (!leaf) return -1;
        // binary search within leaf
        int lo = 0, hi = leaf->num_pairs - 1;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            if (leaf->pairs[mid].first == key) return mid;
            if (leaf->pairs[mid].first < key) lo = mid + 1;
            else hi = mid - 1;
        }
        return -1;
    }

    int find_with_cost(key_T key) {
        LeafNode* leaf = search_leaf_cost(key);
        if (!leaf) return -1;
        int lo = 0, hi = leaf->num_pairs - 1;
        while (lo <= hi) {
            ++Hits::leaf_cost;
            Hits::leaf_max_cost = std::max<long long>(Hits::leaf_max_cost, 1);
            int mid = (lo + hi) >> 1;
            if (leaf->pairs[mid].first == key) return mid;
            if (leaf->pairs[mid].first < key) lo = mid + 1;
            else hi = mid - 1;
        }
        return -1;
    }

    bool get(key_T key, value_T& out) const {
        LeafNode* leaf = search_leaf(key);
        if (!leaf) return false;
        int lo = 0, hi = leaf->num_pairs - 1;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            if (leaf->pairs[mid].first == key) { out = leaf->pairs[mid].second; return true; }
            if (leaf->pairs[mid].first < key) lo = mid + 1;
            else hi = mid - 1;
        }
        return false;
    }

    // ── bulk load (bottom-up from sorted data, O(n)) ────────

    template<class Iterator>
    void bulk_load(Iterator begin, Iterator end) {
        size_t n = std::distance(begin, end);
        size = (int)n;
        if (n == 0) return;

        // Phase 1: create leaf nodes, fill to ~ORDER
        std::vector<LeafNode*> leaves;
        auto it = begin;
        while (it != end) {
            LeafNode* leaf = make_leaf();
            int cnt = 0;
            while (cnt < ORDER && it != end) {
                leaf->pairs[cnt] = *it;
                ++cnt; ++it;
            }
            leaf->num_pairs = cnt;
            leaves.push_back(leaf);
        }
        // link leaves
        for (size_t i = 0; i + 1 < leaves.size(); ++i)
            leaves[i]->next = leaves[i + 1];

        // Phase 2: build internal levels bottom-up
        std::vector<void*> children(leaves.begin(), leaves.end());
        root = build_internal_level(children, true);
    }

private:
    // Build one level of internal nodes from a vector of child nodes.
    // Returns the root InternalNode for this (sub)tree.
    static InternalNode* build_internal_level(std::vector<void*>& children, bool leaf_children) {
        if (children.size() == 1 && leaf_children) {
            // Single leaf — wrap in internal node for uniform root
            InternalNode* in = make_internal();
            in->leaf_children = true;
            in->children[0] = children[0];
            in->num_keys = 0;
            return in;
        }

        std::vector<void*> next_level;
        size_t i = 0;
        while (i < children.size()) {
            InternalNode* in = make_internal();
            in->leaf_children = leaf_children;
            int cnt = 0;
            key_T max_k = 0;
            while (cnt < ORDER + 1 && i < children.size()) {
                // compute separator from max key in the child being added
                key_T child_max = leaf_children
                    ? leaf_max_key((LeafNode*) children[i])
                    : subtree_max_key((InternalNode*) children[i]);
                if (cnt > 0) in->keys[cnt - 1] = max_k;
                in->children[cnt] = children[i];
                max_k = child_max;
                ++cnt; ++i;
            }
            in->num_keys = cnt - 1;
            next_level.push_back(in);
        }

        if (next_level.size() == 1)
            return (InternalNode*) next_level[0];
        return build_internal_level(next_level, false);
    }

    static key_T leaf_max_key(LeafNode* leaf) {
        return leaf->num_pairs > 0 ? leaf->pairs[leaf->num_pairs - 1].first : 0;
    }

    static key_T subtree_max_key(InternalNode* node) {
        // descend rightmost path to leaf
        InternalNode* cur = node;
        while (!cur->leaf_children)
            cur = (InternalNode*) cur->children[cur->num_keys];
        return leaf_max_key((LeafNode*) cur->children[cur->num_keys]);
    }

public:
    // ── single insert (online use, with splits) ─────────────

    bool insert(key_T key, value_T value) {
        if (!root) {
            // First insert: create leaf wrapped in internal node
            LeafNode* leaf = make_leaf();
            leaf->pairs[0] = {key, value};
            leaf->num_pairs = 1;
            root = make_internal();
            root->leaf_children = true;
            root->children[0] = leaf;
            root->num_keys = 0;
            size = 1;
            return true;
        }

        LeafNode* leaf = search_leaf(key);

        // Check duplicate — overwrite
        for (int i = 0; i < leaf->num_pairs; ++i) {
            if (leaf->pairs[i].first == key) {
                leaf->pairs[i].second = value;
                return true;
            }
        }

        // Insert in sorted position
        int pos = 0;
        while (pos < leaf->num_pairs && leaf->pairs[pos].first < key) ++pos;
        for (int i = leaf->num_pairs; i > pos; --i)
            leaf->pairs[i] = leaf->pairs[i - 1];
        leaf->pairs[pos] = {key, value};
        leaf->num_pairs++;
        size++;

        // Split if overflow
        if (leaf->num_pairs >= ORDER) {
            split_leaf(leaf);
        }
        return true;
    }

private:
    void split_leaf(LeafNode* left) {
        LeafNode* right = make_leaf();
        int mid = left->num_pairs / 2;
        for (int i = mid; i < left->num_pairs; ++i)
            right->pairs[i - mid] = left->pairs[i];
        right->num_pairs = left->num_pairs - mid;
        left->num_pairs = mid;
        right->next = left->next;
        left->next = right;

        key_T sep = right->pairs[0].first;
        insert_in_parent(left, sep, right);
    }

    // Rebuild path from root to find the parent of `child`
    void insert_in_parent(void* child, key_T sep, void* new_child) {
        // Walk down from root tracking parent stack
        std::vector<InternalNode*> stack;
        InternalNode* cur = root;
        while (cur && !cur->leaf_children) {
            stack.push_back(cur);
            // find which child to follow
            int pos = 0;
            int lo = 0, hi = cur->num_keys - 1;
            while (lo <= hi) {
                int mid = (lo + hi) >> 1;
                if (sep < cur->keys[mid]) hi = mid - 1;
                else { lo = mid + 1; pos = lo; }
            }
            cur = (InternalNode*) cur->children[pos];
        }

        if (stack.empty()) {
            // Root is parent of leaves — split root if needed
            insert_into_node(root, child, sep, new_child);
            return;
        }

        // Insert into direct parent (last internal node before leaves)
        InternalNode* parent = stack.back();
        InternalNode* new_parent = insert_into_node(parent, child, sep, new_child);

        // Propagate splits upward
        for (int i = (int)stack.size() - 1; i >= 0 && new_parent; --i) {
            InternalNode* node = stack[i];
            // find separator for new_parent and insert into node's parent
            key_T up_sep = subtree_max_key(node); // approximate — the right child's min
            if (i > 0) {
                new_parent = insert_into_node(stack[i - 1], node, up_sep, new_parent);
            } else {
                // Root was split — create new root
                InternalNode* new_root = make_internal();
                new_root->leaf_children = false;
                new_root->keys[0] = up_sep;
                new_root->children[0] = node;
                new_root->children[1] = new_parent;
                new_root->num_keys = 1;
                root = new_root;
                return;
            }
        }
    }

    // Insert (sep, new_child) into node right after `child`.
    // Returns new node if node was split, nullptr otherwise.
    InternalNode* insert_into_node(InternalNode* node, void* child, key_T sep, void* new_child) {
        // find child's position
        int pos = -1;
        for (int i = 0; i <= node->num_keys; ++i) {
            if (node->children[i] == child) { pos = i; break; }
        }
        if (pos < 0) return nullptr;  // shouldn't happen

        // shift to make room
        for (int i = node->num_keys; i > pos; --i)
            node->keys[i] = node->keys[i - 1];
        node->keys[pos] = sep;
        for (int i = node->num_keys + 1; i > pos + 1; --i)
            node->children[i] = node->children[i - 1];
        node->children[pos + 1] = new_child;
        node->num_keys++;

        if (node->num_keys >= ORDER) {
            return split_internal(node);
        }
        return nullptr;
    }

    InternalNode* split_internal(InternalNode* left) {
        InternalNode* right = make_internal();
        right->leaf_children = left->leaf_children;
        int mid = left->num_keys / 2;
        // mid key goes up to parent
        for (int i = mid + 1; i < left->num_keys; ++i)
            right->keys[i - mid - 1] = left->keys[i];
        for (int i = mid + 1; i <= left->num_keys; ++i)
            right->children[i - mid - 1] = left->children[i];
        right->num_keys = left->num_keys - mid - 1;
        left->num_keys = mid;
        return right;
    }

public:
    // ── erase ───────────────────────────────────────────────

    bool erase(key_T key) {
        LeafNode* leaf = search_leaf(key);
        if (!leaf) return false;
        for (int i = 0; i < leaf->num_pairs; ++i) {
            if (leaf->pairs[i].first == key) {
                for (int j = i; j < leaf->num_pairs - 1; ++j)
                    leaf->pairs[j] = leaf->pairs[j + 1];
                leaf->num_pairs--;
                size--;
                return true;
            }
        }
        return false;
    }

    // ── iteration ───────────────────────────────────────────

    LeafNode* first_leaf() const {
        if (!root) return nullptr;
        InternalNode* cur = root;
        while (!cur->leaf_children)
            cur = (InternalNode*) cur->children[0];
        return (LeafNode*) cur->children[0];
    }

    size_t memory_bytes() const {
        // Total memory tracked via node_memory_count — return current delta
        // Actually, this returns total bytes allocated for this tree.
        // For simplicity, count nodes by traversing.
        return count_node_bytes(root);
    }

private:
    static size_t count_node_bytes(InternalNode* node) {
        if (!node) return 0;
        size_t bytes = sizeof(InternalNode);
        if (node->leaf_children) {
            for (int i = 0; i <= node->num_keys; ++i)
                if (node->children[i]) bytes += sizeof(LeafNode);
        } else {
            for (int i = 0; i <= node->num_keys; ++i)
                bytes += count_node_bytes((InternalNode*) node->children[i]);
        }
        return bytes;
    }
};

#endif // BPLUSTREE_H
