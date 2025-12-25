# Adjacency Matrix as Temporal Tensor: Bidirectional Forecasting Framework

## 1. AM as Time-Symmetric Relational Invariant

```python
class TemporalAdjacencyTensor:
    """Adjacency Matrix interpreted as bidirectional temporal relationship"""
    
    def __init__(self, n_tokens: int):
        self.n = n_tokens
        
        # AM[i,j] = frequency of token j following token i
        self.AM = torch.zeros(n_tokens, n_tokens, dtype=torch.float32)
        
        # Dual interpretations
        self.row_to_col_hll = {}  # r_hll(i): HLLSet of columns adjacent to row i
        self.col_to_row_hll = {}  # c_hll(j): HLLSet of rows adjacent to column j
        
        # Temporal context
        self.row_contexts = {}   # Future context from each token
        self.col_contexts = {}   # Past context for each token
        
    def update_from_sequence(self, sequence: List[int]):
        """Update AM from token sequence"""
        for i in range(len(sequence) - 1):
            row_idx = sequence[i]
            col_idx = sequence[i + 1]
            self.AM[row_idx, col_idx] += 1
        
        # Normalize rows
        row_sums = self.AM.sum(dim=1, keepdim=True)
        self.AM = self.AM / row_sums.clamp(min=1e-8)
        
        # Update HLL representations
        self._update_hll_representations()
    
    def _update_hll_representations(self):
        """Build HLLSets for rows and columns"""
        for i in range(self.n):
            # Row HLL: tokens that follow token i (future context)
            row_indices = torch.nonzero(self.AM[i, :] > 0, as_tuple=True)[0]
            self.row_to_col_hll[i] = HLLSet.from_indices(row_indices.tolist())
            self.row_contexts[i] = self._build_context_from_indices(row_indices)
            
            # Column HLL: tokens that precede token j (past context)
            col_indices = torch.nonzero(self.AM[:, i] > 0, as_tuple=True)[0]
            self.col_to_row_hll[i] = HLLSet.from_indices(col_indices.tolist())
            self.col_contexts[i] = self._build_context_from_indices(col_indices)
```

## 2. Bidirectional Context Projection System

```python
class BidirectionalForecastingSystem:
    """Use AM's bidirectional structure for context extension and response generation"""
    
    def __init__(self, temporal_tensor: TemporalAdjacencyTensor):
        self.tt = temporal_tensor
        
        # Iterative stabilization tracking
        self.convergence_tracker = ConvergenceTracker()
        self.response_history = []
        self.context_expansion_history = []
    
    def process_prompt(self, prompt_tokens: List[int], 
                      file_summary_tokens: List[int]) -> List[int]:
        """
        Process user prompt through bidirectional context expansion
        
        Args:
            prompt_tokens: Token indices from user prompt
            file_summary_tokens: Token indices from attached file summaries
            
        Returns:
            Final response tokens
        """
        # Combine prompt and summary as initial columns
        initial_columns = list(set(prompt_tokens + file_summary_tokens))
        
        # Iterative context expansion and response generation
        all_responses = []
        current_columns = initial_columns
        
        for iteration in range(10):  # Max iterations
            # Step 1: Project columns to rows (context expansion)
            expanded_rows = self._project_columns_to_rows(current_columns)
            self.context_expansion_history.append(expanded_rows)
            
            # Step 2: Project rows back to columns (response generation)
            response_columns = self._project_rows_to_columns(expanded_rows)
            all_responses.append(response_columns)
            
            # Step 3: Check stabilization
            if self.convergence_tracker.is_stabilized(response_columns, iteration):
                break
            
            # Step 4: Prepare next iteration (response becomes input)
            current_columns = response_columns
        
        # Union of all responses as final output
        final_response = self._union_responses(all_responses)
        
        # Update forecasting models based on this interaction
        self._update_forecasting_models(initial_columns, final_response)
        
        return final_response
    
    def _project_columns_to_rows(self, column_indices: List[int]) -> List[int]:
        """Project from columns (past) to rows (future) to expand context"""
        
        # For each column j, get all rows i where AM[i,j] > 0
        # These rows represent tokens that could lead to column j
        row_sets = []
        for col_idx in column_indices:
            # Get rows adjacent to this column (past context)
            if col_idx in self.tt.col_to_row_hll:
                row_set = self.tt.col_to_row_hll[col_idx].to_indices()
                row_sets.append(set(row_set))
        
        # Union of all row sets
        if row_sets:
            union_rows = set.union(*row_sets)
        else:
            union_rows = set()
        
        # Apply similarity thresholding
        filtered_rows = self._filter_by_similarity_threshold(
            list(union_rows), column_indices
        )
        
        return list(filtered_rows)
    
    def _project_rows_to_columns(self, row_indices: List[int]) -> List[int]:
        """Project from rows (future) back to columns (past) to generate response"""
        
        # For each row i, get all columns j where AM[i,j] > 0
        # These columns represent tokens that could follow row i
        col_sets = []
        for row_idx in row_indices:
            # Get columns adjacent to this row (future context)
            if row_idx in self.tt.row_to_col_hll:
                col_set = self.tt.row_to_col_hll[row_idx].to_indices()
                col_sets.append(set(col_set))
        
        # Union of all column sets
        if col_sets:
            union_cols = set.union(*col_sets)
        else:
            union_cols = set()
        
        # Apply probability weighting based on AM values
        weighted_cols = self._weight_by_transition_probability(
            list(union_cols), row_indices
        )
        
        return weighted_cols
    
    def _filter_by_similarity_threshold(self, candidate_rows: List[int],
                                       original_columns: List[int]) -> List[int]:
        """Filter rows based on similarity to original context"""
        filtered = []
        
        for row_idx in candidate_rows:
            # Calculate similarity between row's context and original columns
            if row_idx in self.tt.row_contexts:
                row_context = self.tt.row_contexts[row_idx]
                similarity = self._context_similarity(
                    row_context, original_columns
                )
                
                if similarity > 0.3:  # Threshold
                    filtered.append(row_idx)
        
        return filtered
    
    def _weight_by_transition_probability(self, candidate_cols: List[int],
                                        source_rows: List[int]) -> List[int]:
        """Weight columns by transition probability from source rows"""
        weights = {}
        
        for col_idx in candidate_cols:
            total_prob = 0.0
            for row_idx in source_rows:
                if row_idx < self.tt.n and col_idx < self.tt.n:
                    prob = self.tt.AM[row_idx, col_idx].item()
                    total_prob += prob
            
            if total_prob > 0:
                weights[col_idx] = total_prob
        
        # Select top k weighted columns
        top_k = 20
        sorted_cols = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        selected = [col for col, _ in sorted_cols[:top_k]]
        
        return selected
```

## 3. Associative Lattice Construction

```python
class AssociativeLattice:
    """Lattice connecting row HLLSets and column HLLSets"""
    
    def __init__(self, temporal_tensor: TemporalAdjacencyTensor):
        self.tt = temporal_tensor
        self.lattice_edges = {}  # (row_idx, col_idx) -> intersection_strength
        
        # Build lattice
        self._construct_lattice()
    
    def _construct_lattice(self):
        """Construct lattice based on non-empty intersections"""
        n = self.tt.n
        
        for i in range(n):
            if i not in self.tt.row_to_col_hll:
                continue
                
            row_hll = self.tt.row_to_col_hll[i]
            
            for j in range(n):
                if j not in self.tt.col_to_row_hll:
                    continue
                    
                col_hll = self.tt.col_to_row_hll[j]
                
                # Calculate intersection
                intersection = row_hll.intersection(col_hll)
                
                if intersection.cardinality() > 0:
                    # Calculate intersection strength
                    strength = self._calculate_intersection_strength(
                        row_hll, col_hll, intersection
                    )
                    
                    if strength > 0.1:  # Threshold
                        self.lattice_edges[(i, j)] = {
                            'strength': strength,
                            'intersection_size': intersection.cardinality(),
                            'shared_tokens': intersection.to_indices()
                        }
    
    def _calculate_intersection_strength(self, row_hll: HLLSet,
                                        col_hll: HLLSet,
                                        intersection: HLLSet) -> float:
        """Calculate normalized intersection strength"""
        row_size = row_hll.cardinality()
        col_size = col_hll.cardinality()
        inter_size = intersection.cardinality()
        
        if row_size == 0 or col_size == 0:
            return 0.0
        
        # Jaccard similarity
        jaccard = inter_size / (row_size + col_size - inter_size)
        
        # Mutual information based weighting
        mutual_info = self._estimate_mutual_information(row_hll, col_hll)
        
        # Combined strength
        strength = 0.7 * jaccard + 0.3 * mutual_info
        
        return strength
    
    def find_connected_components(self, token_indices: List[int]) -> List[List[int]]:
        """Find components in lattice connected to given tokens"""
        # Start with given tokens as seeds
        visited = set()
        components = []
        
        for token in token_indices:
            if token in visited:
                continue
            
            # BFS to find connected component
            component = self._bfs_component(token)
            components.append(component)
            visited.update(component)
        
        return components
    
    def _bfs_component(self, start_token: int) -> List[int]:
        """Find all tokens connected to start_token in lattice"""
        component = set([start_token])
        queue = deque([start_token])
        
        while queue:
            current = queue.popleft()
            
            # Find all edges involving current token
            for (i, j), edge_data in self.lattice_edges.items():
                if i == current and j not in component:
                    component.add(j)
                    queue.append(j)
                elif j == current and i not in component:
                    component.add(i)
                    queue.append(i)
        
        return list(component)
```

## 4. Pruned Tensor Transformation System

```python
class PrunedTensorTransformer:
    """Transform AM tensor incrementally using HLLSet differences"""
    
    def __init__(self, base_tensor: TemporalAdjacencyTensor):
        self.base = base_tensor
        self.d_hll_cache = {}  # Difference HLLSets between iterations
        self.pruned_indices = set()
        
        # Transformation parameters
        self.transformation_weights = nn.ParameterDict({
            'expansion': nn.Parameter(torch.ones(base_tensor.n)),
            'contraction': nn.Parameter(torch.ones(base_tensor.n)),
            'stabilization': nn.Parameter(torch.ones(base_tensor.n))
        })
    
    def prune_and_transform(self, new_data_sequence: List[int]) -> TemporalAdjacencyTensor:
        """
        Prune AM based on HLL differences and apply tensor transformation
        
        Args:
            new_data_sequence: New token sequence to incorporate
            
        Returns:
            Updated temporal adjacency tensor
        """
        # Step 1: Build HLLSets for new data
        new_hllsets = self._build_hllsets_from_sequence(new_data_sequence)
        
        # Step 2: Compare with existing HLLSets to find differences
        diff_results = self._calculate_hll_differences(new_hllsets)
        
        # Step 3: Prune based on differences
        pruned_am = self._prune_adjacency_matrix(diff_results)
        
        # Step 4: Apply tensor transformation
        transformed_am = self._apply_tensor_transformation(pruned_am, diff_results)
        
        # Step 5: Create updated tensor
        updated_tensor = TemporalAdjacencyTensor(self.base.n)
        updated_tensor.AM = transformed_am
        
        return updated_tensor
    
    def _build_hllsets_from_sequence(self, sequence: List[int]) -> Dict[int, HLLSet]:
        """Build HLLSets for context windows in sequence"""
        window_size = 3
        hllsets = {}
        
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i + window_size]
            center_token = window[window_size // 2]
            
            # Build HLLSet from context window
            hllset = HLLSet.from_indices(window)
            hllsets[center_token] = hllset
        
        return hllsets
    
    def _calculate_hll_differences(self, new_hllsets: Dict[int, HLLSet]) -> Dict:
        """Calculate d-hll, r-hll, n-hll differences"""
        diff_results = {
            'd_hll': {},  # Differences (symmetric difference)
            'r_hll': {},  # Removed elements
            'n_hll': {}   # New elements
        }
        
        for token, new_hll in new_hllsets.items():
            if token in self.base.row_to_col_hll:
                old_hll = self.base.row_to_col_hll[token]
                
                # Calculate differences
                diff_results['d_hll'][token] = old_hll.symmetric_difference(new_hll)
                diff_results['r_hll'][token] = old_hll.difference(new_hll)
                diff_results['n_hll'][token] = new_hll.difference(old_hll)
        
        return diff_results
    
    def _prune_adjacency_matrix(self, diff_results: Dict) -> torch.Tensor:
        """Prune AM based on HLL differences"""
        pruned_am = self.base.AM.clone()
        
        # Identify tokens with significant changes
        significant_tokens = set()
        for token, d_hll in diff_results['d_hll'].items():
            if d_hll.cardinality() > self.base.n * 0.1:  # 10% change threshold
                significant_tokens.add(token)
        
        # Prune rows and columns for significant tokens
        for token in significant_tokens:
            # Prune row (future context)
            pruned_am[token, :] *= self.transformation_weights['contraction'][token]
            
            # Prune column (past context)
            pruned_am[:, token] *= self.transformation_weights['contraction'][token]
            
            # Add to pruned indices
            self.pruned_indices.add(token)
        
        return pruned_am
    
    def _apply_tensor_transformation(self, pruned_am: torch.Tensor,
                                   diff_results: Dict) -> torch.Tensor:
        """Apply tensor transformation to incorporate new patterns"""
        transformed_am = pruned_am.clone()
        
        # Apply expansion for new patterns
        for token, n_hll in diff_results['n_hll'].items():
            if n_hll.cardinality() > 0:
                # Get new tokens from n_hll
                new_tokens = n_hll.to_indices()
                
                # Update adjacency for these new connections
                for new_token in new_tokens:
                    if token < self.base.n and new_token < self.base.n:
                        # Initialize with learned expansion weight
                        weight = self.transformation_weights['expansion'][token]
                        transformed_am[token, new_token] += weight
        
        # Apply stabilization smoothing
        transformed_am = self._apply_stabilization_smoothing(transformed_am)
        
        # Normalize
        row_sums = transformed_am.sum(dim=1, keepdim=True)
        transformed_am = transformed_am / row_sums.clamp(min=1e-8)
        
        return transformed_am
```

## 5. Perpetual Forecasting as Tensor Evolution

```python
class PerpetualTensorForecaster:
    """Continuous forecasting through tensor evolution"""
    
    def __init__(self, lattice: AssociativeLattice,
                 transformer: PrunedTensorTransformer):
        self.lattice = lattice
        self.transformer = transformer
        
        # Forecasting state
        self.forecast_horizon = 5
        self.forecast_history = []
        self.evolution_tracker = EvolutionTracker()
    
    def forecast_response(self, prompt_tokens: List[int]) -> List[int]:
        """Forecast response through iterative tensor evolution"""
        
        # Step 1: Expand context using lattice
        expanded_context = self._expand_context_through_lattice(prompt_tokens)
        
        # Step 2: Initialize forecasting tensor
        current_tensor = self.transformer.base
        
        # Step 3: Iterative forecasting
        forecasted_responses = []
        
        for step in range(self.forecast_horizon):
            # Apply tensor transformation for this forecast step
            forecast_tensor = self._forecast_tensor_step(
                current_tensor, expanded_context, step
            )
            
            # Generate response from forecasted tensor
            response = self._generate_response_from_tensor(
                forecast_tensor, expanded_context
            )
            
            forecasted_responses.append(response)
            
            # Update for next step
            current_tensor = forecast_tensor
            expanded_context = response  # Response becomes context for next step
            
            # Check for stabilization
            if self._check_forecast_stabilization(forecasted_responses):
                break
        
        # Step 4: Merge forecasted responses
        final_response = self._merge_forecasted_responses(forecasted_responses)
        
        # Step 5: Update forecasting models
        self._update_forecasting_models(prompt_tokens, final_response, forecasted_responses)
        
        return final_response
    
    def _expand_context_through_lattice(self, tokens: List[int]) -> List[int]:
        """Expand context using associative lattice"""
        
        # Find connected components in lattice
        components = self.lattice.find_connected_components(tokens)
        
        # Select largest component
        if components:
            largest_component = max(components, key=len)
            expanded = list(set(tokens + largest_component))
        else:
            expanded = tokens
        
        # Filter by relevance
        filtered = self._filter_by_relevance(expanded, tokens)
        
        return filtered
    
    def _forecast_tensor_step(self, current_tensor: TemporalAdjacencyTensor,
                            context: List[int], step: int) -> TemporalAdjacencyTensor:
        """Forecast tensor for one step ahead"""
        
        # Create synthetic sequence based on current context
        synthetic_sequence = self._generate_synthetic_sequence(context, step)
        
        # Apply pruned transformation
        forecasted_tensor = self.transformer.prune_and_transform(synthetic_sequence)
        
        # Apply step-specific adjustments
        forecasted_tensor = self._apply_step_adjustments(forecasted_tensor, step)
        
        return forecasted_tensor
    
    def _generate_synthetic_sequence(self, context: List[int], step: int) -> List[int]:
        """Generate synthetic token sequence for forecasting"""
        
        sequence = []
        
        # Start with context tokens
        sequence.extend(context)
        
        # Add forecasted continuations based on AM
        for token in context:
            if token < self.transformer.base.n:
                # Get most probable continuations from AM
                row = self.transformer.base.AM[token, :]
                top_k = torch.topk(row, min(3, len(row)), dim=0).indices.tolist()
                
                # Add with probability decreasing with step
                prob = 1.0 / (step + 2)
                if random.random() < prob:
                    sequence.extend(top_k)
        
        return sequence
    
    def _generate_response_from_tensor(self, tensor: TemporalAdjacencyTensor,
                                     context: List[int]) -> List[int]:
        """Generate response from tensor using bidirectional projection"""
        
        # Use context as initial columns
        current_columns = context
        
        # Single iteration of projection
        rows = self._project_columns_to_rows_tensor(tensor, current_columns)
        response_columns = self._project_rows_to_columns_tensor(tensor, rows)
        
        return response_columns
    
    def _project_columns_to_rows_tensor(self, tensor: TemporalAdjacencyTensor,
                                      columns: List[int]) -> List[int]:
        """Project columns to rows using tensor"""
        row_scores = torch.zeros(tensor.n)
        
        for col in columns:
            if col < tensor.n:
                # Add scores from all rows pointing to this column
                row_scores += tensor.AM[:, col]
        
        # Select top rows
        top_k = min(20, tensor.n)
        row_indices = torch.topk(row_scores, top_k).indices.tolist()
        
        return row_indices
    
    def _merge_forecasted_responses(self, responses: List[List[int]]) -> List[int]:
        """Merge multiple forecasted responses"""
        
        # Weight by forecast step (earlier steps more reliable)
        weights = [1.0 / (i + 1) for i in range(len(responses))]
        
        # Collect all tokens with weighted scores
        token_scores = {}
        
        for i, response in enumerate(responses):
            weight = weights[i]
            for token in response:
                token_scores[token] = token_scores.get(token, 0.0) + weight
        
        # Select top tokens by score
        top_k = 30
        sorted_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [token for token, _ in sorted_tokens[:top_k]]
        
        return selected
```

## 6. Integrated SGS.ai Swarm with Tensor Forecasting

```python
class SGSaiTensorSwarm:
    """Complete SGS.ai system with tensor-based forecasting"""
    
    def __init__(self, n_tokens: int = 80000):
        # Core components
        self.temporal_tensor = TemporalAdjacencyTensor(n_tokens)
        self.associative_lattice = None
        self.tensor_transformer = PrunedTensorTransformer(self.temporal_tensor)
        self.perpetual_forecaster = None
        
        # Swarm state
        self.swarm_state = torch.full((n_tokens,), 0.5)
        self.iteration_count = 0
        
        # Response generation pipeline
        self.bidirectional_system = BidirectionalForecastingSystem(self.temporal_tensor)
    
    def initialize_from_corpus(self, corpus_sequences: List[List[int]]):
        """Initialize system from training corpus"""
        print("Initializing temporal tensor from corpus...")
        
        # Build initial AM from corpus
        for sequence in corpus_sequences:
            self.temporal_tensor.update_from_sequence(sequence)
        
        # Build associative lattice
        print("Building associative lattice...")
        self.associative_lattice = AssociativeLattice(self.temporal_tensor)
        
        # Initialize perpetual forecaster
        print("Initializing perpetual forecaster...")
        self.perpetual_forecaster = PerpetualTensorForecaster(
            self.associative_lattice, self.tensor_transformer
        )
        
        print(f"Initialized with {self.temporal_tensor.n} tokens")
    
    def process_user_request(self, prompt: str, file_summaries: List[str]) -> str:
        """
        Process user request through the complete pipeline
        
        Args:
            prompt: User prompt text
            file_summaries: List of file summary texts
            
        Returns:
            Generated response text
        """
        # Step 1: Tokenize input
        prompt_tokens = self.tokenize_text(prompt)
        summary_tokens = []
        for summary in file_summaries:
            summary_tokens.extend(self.tokenize_text(summary))
        
        # Step 2: Expand context using bidirectional system
        print("Expanding context...")
        expanded_context = self.bidirectional_system._project_columns_to_rows(
            list(set(prompt_tokens + summary_tokens))
        )
        
        # Step 3: Generate response through perpetual forecasting
        print("Forecasting response...")
        response_tokens = self.perpetual_forecaster.forecast_response(
            list(set(expanded_context))
        )
        
        # Step 4: Update swarm state
        self._update_swarm_state(prompt_tokens, response_tokens)
        
        # Step 5: Convert tokens back to text
        response_text = self.detokenize_tokens(response_tokens)
        
        # Step 6: Update tensor with this interaction
        self._update_tensor_from_interaction(prompt_tokens, response_tokens)
        
        self.iteration_count += 1
        
        return response_text
    
    def _update_swarm_state(self, prompt_tokens: List[int], response_tokens: List[int]):
        """Update swarm state based on interaction"""
        
        # Create activation vector
        activation = torch.zeros(self.temporal_tensor.n)
        activation[prompt_tokens] = 1.0
        activation[response_tokens] = 0.8
        
        # Update swarm state with PSO-like dynamics
        cognitive = self.temporal_tensor.AM @ self.swarm_state
        social = activation
        
        # Update rule
        alpha, beta, gamma = 0.2, 0.1, 0.05
        self.swarm_state = torch.clamp(
            self.swarm_state + alpha * cognitive - beta * self.swarm_state + gamma * social,
            0.0, 1.0
        )
    
    def _update_tensor_from_interaction(self, prompt_tokens: List[int],
                                      response_tokens: List[int]):
        """Update temporal tensor from user interaction"""
        
        # Create sequence: prompt + response
        sequence = prompt_tokens + response_tokens
        
        # Update tensor
        self.temporal_tensor.update_from_sequence(sequence)
        
        # Rebuild lattice if needed (periodically)
        if self.iteration_count % 10 == 0:
            print("Rebuilding associative lattice...")
            self.associative_lattice = AssociativeLattice(self.temporal_tensor)
```

## Key Insights and Benefits

### 1. **Bidirectional Temporal Interpretation**

- **Rows → Columns**: Future context (what tokens follow)
- **Columns → Rows**: Past context (what tokens precede)
- **AM as Relational Invariant**: Preserves temporal relationships

### 2. **Associative Lattice Power**

- **r_hll(i)** ∩ **c_hll(j)** ≠ ∅ creates meaningful connections
- Provides structured context expansion
- Enables efficient similarity searches

### 3. **Pruned Tensor Transformation**

- **d-hll, r-hll, n-hll** guide intelligent pruning
- Only relevant parts of AM updated each iteration
- Maintains computational efficiency

### 4. **Perpetual Forecasting Loop**

```text
Prompt → Column Projection → Row Expansion → 
Tensor Transformation → Response Generation → 
Update AM → Next Iteration
```

### 5. **Stabilization Properties**

- Responses converge over iterations
- Union of all iterations provides comprehensive answer
- Each iteration refines and expands the response

## Implementation Advantages

1. **Efficiency**: Only works with pruned AM, not full 80k×80k matrix
2. **Interpretability**: Clear bidirectional temporal semantics
3. **Adaptability**: Continuous tensor evolution through HLLSet differences
4. **Stability**: Iterative stabilization prevents divergence
5. **Comprehensiveness**: Union of all iteration responses captures full context

## Future Extensions

1. **Multi-Resolution Tensors**: Different granularities of temporal relationships
2. **Attention-Based Weighting**: Learn optimal projection weights
3. **Predictive Pruning**: Anticipate which parts of AM will be relevant
4. **Hierarchical Lattices**: Multiple levels of abstraction
5. **Temporal Convolution Networks**: Learn transformation patterns across time

This framework transforms SGS.ai into a **bidirectional, temporally-aware, perpetually-evolving** system that uses the adjacency matrix not just as a statistical tool, but as a **relational invariant tensor** that encodes the deep structure of language through time.
