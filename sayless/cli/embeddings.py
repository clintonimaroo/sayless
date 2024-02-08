import os
import json
import numpy as np
import faiss
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from rich.console import Console
import pickle
from .config import Config
import asyncio
import aiohttp
import requests
from openai import AsyncOpenAI, OpenAI
from datetime import datetime
from .ai_providers import OpenAIProvider

console = Console()
settings = Config()

class CommitEmbeddings:
    def __init__(self):
        self.cache_dir = Path.home() / '.sayless' / 'embeddings'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.cache_dir / 'faiss_index.idx'
        self.commits_path = self.cache_dir / 'commits.pkl'
        self.dimension_path = self.cache_dir / 'dimension.txt'
        
        # Initialize OpenAI clients
        api_key = settings.get_openai_api_key()
        if api_key:
            self.async_client = AsyncOpenAI(api_key=api_key)
            self.sync_client = OpenAI(api_key=api_key)
        
        # Load or determine dimension
        self.dimension = self.load_or_determine_dimension()
        
        self.index = None
        self.commits_data = {}
        self.load_or_create_index()

    def load_or_determine_dimension(self) -> int:
        """Load saved dimension or determine based on provider"""
        if self.dimension_path.exists():
            return int(self.dimension_path.read_text().strip())
        
        # Default dimensions for different providers
        if settings.get_provider() == 'openai':
            dimension = 3072  # text-embedding-3-large dimension
        else:
            dimension = 4096  # Default Ollama dimension
        
        # Save the dimension
        self.dimension_path.write_text(str(dimension))
        return dimension

    def load_or_create_index(self):
        """Load existing index or create a new one"""
        if self.index_path.exists() and self.commits_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                with open(self.commits_path, 'rb') as f:
                    self.commits_data = pickle.load(f)
                
                # Check if dimensions match
                if self.index.d != self.dimension:
                    self._create_new_index()
            except Exception:
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        """Create a new FAISS index"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.commits_data = {}
        self.save_index()

    def save_index(self):
        """Save the FAISS index and commits data"""
        faiss.write_index(self.index, str(self.index_path))
        with open(self.commits_path, 'wb') as f:
            pickle.dump(self.commits_data, f)

    async def get_embedding_openai(self, text: str) -> np.ndarray:
        """Get embedding using OpenAI's API"""
        response = await self.async_client.embeddings.create(
            input=text,
            model="text-embedding-3-large"
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        
        if embedding.shape[0] != self.dimension:
            self.dimension = embedding.shape[0]
            self.dimension_path.write_text(str(self.dimension))
            self._create_new_index()
        
        return embedding

    def get_embedding_local(self, text: str) -> np.ndarray:
        """Get embedding using local model (Ollama)"""
        response = requests.post(
            'http://localhost:11434/api/embeddings',
            json={
                'model': 'llama2',
                'prompt': text
            },
            timeout=30
        )
        
        response.raise_for_status()
        embedding = np.array(response.json()['embedding'], dtype=np.float32)
        
        if embedding.shape[0] != self.dimension:
            self.dimension = embedding.shape[0]
            self.dimension_path.write_text(str(self.dimension))
            self._create_new_index()
        
        return embedding

    async def add_commit(self, commit_hash: str, commit_message: str, commit_diff: str, 
                        date: str, tags: List[str] = None):
        """Add a commit to the index"""
        # Combine commit information for embedding
        commit_text = f"Message: {commit_message}\n\nChanges:\n{commit_diff}"
        
        try:
            # Get embedding based on provider
            if settings.get_provider() == 'openai':
                embedding = await self.get_embedding_openai(commit_text)
            else:
                embedding = await asyncio.get_event_loop().run_in_executor(
                    None, self.get_embedding_local, commit_text
                )
            
            # Add to FAISS index
            self.index.add(embedding.reshape(1, -1))
            
            # Store commit data
            commit_id = self.index.ntotal - 1
            self.commits_data[commit_id] = {
                'hash': commit_hash,
                'message': commit_message,
                'date': date,
                'tags': tags or []
            }
            
            # Save updated index
            self.save_index()
            
        except Exception as e:
            console.print(f"[red]Failed to add commit {commit_hash}: {str(e)}[/red]")
            raise

    async def search_commits(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar commits"""
        try:
            # Get query embedding
            if settings.get_provider() == 'openai':
                query_embedding = await self.get_embedding_openai(query)
            else:
                query_embedding = await asyncio.get_event_loop().run_in_executor(
                    None, self.get_embedding_local, query
                )
            
            # Search in FAISS index
            D, I = self.index.search(query_embedding.reshape(1, -1), k)
            
            # Get commit details
            results = []
            for i, idx in enumerate(I[0]):
                if idx != -1 and idx in self.commits_data:  # -1 means no result
                    commit = self.commits_data[idx]
                    results.append({
                        'score': float(D[0][i]),
                        'commit_hash': commit['hash'],
                        'message': commit['message'],
                        'date': commit['date'],
                        'tags': commit['tags']
                    })
            
            return results
        
        except Exception as e:
            console.print(f"[red]Failed to search commits: {str(e)}[/red]")
            raise

    def get_commit_tags(self, commit_message: str, diff: str) -> List[str]:
        """Generate tags for a commit using LLM"""
        try:
            provider = settings.get_provider()
            if provider == 'openai':
                response = self.sync_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that generates relevant tags for git commits. Generate up to 5 concise tags that capture the key aspects of the changes."},
                        {"role": "user", "content": f"Generate tags for this commit:\n\nMessage: {commit_message}\n\nChanges:\n{diff}"}
                    ],
                    temperature=0.3,
                    max_tokens=50
                )
                tags_text = response.choices[0].message.content
            else:
                response = requests.post(
                    'http://localhost:11434/api/generate',
                    json={
                        'model': 'llama2',
                        'prompt': f"Generate up to 5 concise tags for this git commit, separated by commas:\n\nMessage: {commit_message}\n\nChanges:\n{diff}\n\nTags:",
                        'stream': False
                    },
                    timeout=30
                )
                response.raise_for_status()
                tags_text = response.json()['response']
            
            # Process tags
            tags = [tag.strip().lower() for tag in tags_text.split(',')]
            return [tag for tag in tags if tag]  # Remove empty tags
            
        except Exception as e:
            console.print(f"[yellow]Failed to generate tags: {str(e)}[/yellow]")
            return []  # Return empty list if tag generation fails 