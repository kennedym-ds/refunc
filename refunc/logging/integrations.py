"""
External integrations for the logging system.

This module provides integrations with popular third-party services
and libraries for logging, monitoring, and observability.
"""

import os
import sys
import time
import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable, TYPE_CHECKING
from contextlib import contextmanager
import logging

from ..exceptions import RefuncError

if TYPE_CHECKING:
    try:
        import prometheus_client
        import elasticsearch
        import redis
    except ImportError:
        pass


class PrometheusIntegration:
    """Integration with Prometheus for metrics collection."""
    
    def __init__(
        self,
        gateway_url: Optional[str] = None,
        job_name: str = "refunc_ml",
        registry: Optional[Any] = None
    ):
        self.prometheus: Any = None
        self.gateway_url = gateway_url
        self.job_name = job_name
        self.registry = registry
        self.metrics: Dict[str, Any] = {}
        
        try:
            import prometheus_client
            self.prometheus = prometheus_client
            self.available = True
            
            if not self.registry:
                self.registry = prometheus_client.CollectorRegistry()
            
            # Initialize common metrics
            self._init_metrics()
            
        except ImportError:
            self.available = False
    
    def _init_metrics(self) -> None:
        """Initialize common ML metrics."""
        if not self.prometheus:
            return
        
        # Training metrics
        self.metrics['training_loss'] = self.prometheus.Gauge(
            'ml_training_loss',
            'Current training loss',
            registry=self.registry
        )
        
        self.metrics['validation_loss'] = self.prometheus.Gauge(
            'ml_validation_loss',
            'Current validation loss',
            registry=self.registry
        )
        
        self.metrics['training_accuracy'] = self.prometheus.Gauge(
            'ml_training_accuracy',
            'Current training accuracy',
            registry=self.registry
        )
        
        self.metrics['validation_accuracy'] = self.prometheus.Gauge(
            'ml_validation_accuracy',
            'Current validation accuracy',
            registry=self.registry
        )
        
        # System metrics
        self.metrics['epoch_duration'] = self.prometheus.Histogram(
            'ml_epoch_duration_seconds',
            'Time spent per epoch',
            registry=self.registry
        )
        
        self.metrics['batch_processing_time'] = self.prometheus.Histogram(
            'ml_batch_processing_seconds',
            'Time spent processing batches',
            registry=self.registry
        )
        
        # Counters
        self.metrics['epochs_completed'] = self.prometheus.Counter(
            'ml_epochs_completed_total',
            'Total number of completed epochs',
            registry=self.registry
        )
        
        self.metrics['samples_processed'] = self.prometheus.Counter(
            'ml_samples_processed_total',
            'Total number of samples processed',
            registry=self.registry
        )
    
    def log_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Log a metric to Prometheus."""
        if not self.available:
            return
        
        if name in self.metrics:
            metric = self.metrics[name]
            if hasattr(metric, 'set'):
                metric.set(value)
            elif hasattr(metric, 'observe'):
                metric.observe(value)
            elif hasattr(metric, 'inc'):
                metric.inc(value)
        else:
            # Create dynamic gauge
            gauge = self.prometheus.Gauge(
                f'ml_{name}',
                f'ML metric: {name}',
                registry=self.registry
            )
            gauge.set(value)
            self.metrics[name] = gauge
    
    def push_to_gateway(self) -> None:
        """Push metrics to Prometheus pushgateway."""
        if not self.available or not self.gateway_url:
            return
        
        try:
            self.prometheus.push_to_gateway(
                self.gateway_url,
                job=self.job_name,
                registry=self.registry
            )
        except Exception as e:
            logging.warning(f"Failed to push metrics to Prometheus: {e}")


class ElasticsearchIntegration:
    """Integration with Elasticsearch for log aggregation."""
    
    def __init__(
        self,
        hosts: List[str],
        index_name: str = "ml-logs",
        doc_type: str = "_doc"
    ):
        self.elasticsearch: Any = None
        self.hosts = hosts
        self.index_name = index_name
        self.doc_type = doc_type
        self.available = False
        
        try:
            import elasticsearch
            self.elasticsearch = elasticsearch.Elasticsearch(hosts=hosts)
            self.available = True
            
            # Create index if it doesn't exist
            self._create_index()
            
        except ImportError:
            pass
        except Exception as e:
            logging.warning(f"Failed to connect to Elasticsearch: {e}")
    
    def _create_index(self) -> None:
        """Create index with appropriate mapping."""
        if not self.elasticsearch:
            return
        
        mapping = {
            "mappings": {
                "properties": {
                    "timestamp": {"type": "date"},
                    "level": {"type": "keyword"},
                    "message": {"type": "text"},
                    "experiment_id": {"type": "keyword"},
                    "run_id": {"type": "keyword"},
                    "metrics": {"type": "object"},
                    "parameters": {"type": "object"},
                    "tags": {"type": "object"},
                    "hostname": {"type": "keyword"},
                    "python_version": {"type": "keyword"}
                }
            }
        }
        
        try:
            if not self.elasticsearch.indices.exists(index=self.index_name):
                self.elasticsearch.indices.create(
                    index=self.index_name,
                    body=mapping
                )
        except Exception as e:
            logging.warning(f"Failed to create Elasticsearch index: {e}")
    
    def log_entry(self, log_data: Dict[str, Any]) -> None:
        """Send log entry to Elasticsearch."""
        if not self.available:
            return
        
        try:
            # Add timestamp if not present
            if 'timestamp' not in log_data:
                log_data['timestamp'] = time.time() * 1000  # ES expects milliseconds
            
            self.elasticsearch.index(
                index=self.index_name,
                doc_type=self.doc_type,
                body=log_data
            )
        except Exception as e:
            logging.warning(f"Failed to send log to Elasticsearch: {e}")
    
    def search_logs(
        self,
        query: Dict[str, Any],
        size: int = 100,
        sort: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Search logs in Elasticsearch."""
        if not self.available:
            return {"hits": {"hits": []}}
        
        try:
            return self.elasticsearch.search(
                index=self.index_name,
                body={
                    "query": query,
                    "size": size,
                    "sort": sort or [{"timestamp": {"order": "desc"}}]
                }
            )
        except Exception as e:
            logging.warning(f"Failed to search Elasticsearch: {e}")
            return {"hits": {"hits": []}}


class RedisIntegration:
    """Integration with Redis for caching and pub/sub."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "refunc:ml:"
    ):
        self.redis: Any = None
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.prefix = prefix
        self.available = False
        
        try:
            import redis
            self.redis = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True
            )
            
            # Test connection
            self.redis.ping()
            self.available = True
            
        except ImportError:
            pass
        except Exception as e:
            logging.warning(f"Failed to connect to Redis: {e}")
    
    def cache_metrics(self, experiment_id: str, metrics: Dict[str, Any], ttl: int = 3600) -> None:
        """Cache metrics in Redis."""
        if not self.available:
            return
        
        try:
            key = f"{self.prefix}metrics:{experiment_id}"
            self.redis.setex(
                key,
                ttl,
                json.dumps(metrics, default=str)
            )
        except Exception as e:
            logging.warning(f"Failed to cache metrics in Redis: {e}")
    
    def get_cached_metrics(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get cached metrics from Redis."""
        if not self.available:
            return None
        
        try:
            key = f"{self.prefix}metrics:{experiment_id}"
            data = self.redis.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logging.warning(f"Failed to get cached metrics from Redis: {e}")
            return None
    
    def publish_event(self, channel: str, event: Dict[str, Any]) -> None:
        """Publish event to Redis channel."""
        if not self.available:
            return
        
        try:
            self.redis.publish(
                f"{self.prefix}{channel}",
                json.dumps(event, default=str)
            )
        except Exception as e:
            logging.warning(f"Failed to publish event to Redis: {e}")
    
    def subscribe_to_events(self, channels: List[str]) -> Any:
        """Subscribe to Redis channels."""
        if not self.available:
            return None
        
        try:
            pubsub = self.redis.pubsub()
            full_channels = [f"{self.prefix}{ch}" for ch in channels]
            pubsub.subscribe(*full_channels)
            return pubsub
        except Exception as e:
            logging.warning(f"Failed to subscribe to Redis channels: {e}")
            return None


class SlackIntegration:
    """Integration with Slack for notifications."""
    
    def __init__(self, webhook_url: Optional[str] = None, token: Optional[str] = None):
        self.webhook_url = webhook_url
        self.token = token
        self.available = bool(webhook_url or token)
    
    def send_notification(
        self,
        message: str,
        channel: Optional[str] = None,
        level: str = "info"
    ) -> None:
        """Send notification to Slack."""
        if not self.available:
            return
        
        # Color based on level
        color_map = {
            "info": "#36a64f",
            "warning": "#ff9900",
            "error": "#ff0000",
            "success": "#00ff00"
        }
        
        payload = {
            "text": message,
            "attachments": [{
                "color": color_map.get(level, "#36a64f"),
                "fields": [{
                    "title": "Level",
                    "value": level.upper(),
                    "short": True
                }],
                "ts": time.time()
            }]
        }
        
        if channel:
            payload["channel"] = channel
        
        try:
            import requests
            
            if self.webhook_url:
                requests.post(self.webhook_url, json=payload)
            elif self.token:
                # Use Slack API (simplified)
                headers = {"Authorization": f"Bearer {self.token}"}
                requests.post(
                    "https://slack.com/api/chat.postMessage",
                    headers=headers,
                    json=payload
                )
        except Exception as e:
            logging.warning(f"Failed to send Slack notification: {e}")


class DiscordIntegration:
    """Integration with Discord for notifications."""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.available = bool(webhook_url)
    
    def send_notification(self, message: str, level: str = "info") -> None:
        """Send notification to Discord."""
        if not self.available:
            return
        
        # Emoji based on level
        emoji_map = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "success": "✅"
        }
        
        payload = {
            "content": f"{emoji_map.get(level, 'ℹ️')} {message}",
            "embeds": [{
                "title": f"{level.upper()} Notification",
                "description": message,
                "color": {
                    "info": 3447003,
                    "warning": 16776960,
                    "error": 15158332,
                    "success": 3066993
                }.get(level, 3447003),
                "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            }]
        }
        
        try:
            import requests
            requests.post(self.webhook_url, json=payload)
        except Exception as e:
            logging.warning(f"Failed to send Discord notification: {e}")


class IntegrationsManager:
    """
    Central manager for all external integrations.
    
    Manages multiple integrations and provides a unified interface
    for sending data to various external services.
    """
    
    def __init__(self):
        self.integrations: Dict[str, Any] = {}
        self._lock = threading.Lock()
    
    def add_prometheus(self, **kwargs) -> None:
        """Add Prometheus integration."""
        self.integrations['prometheus'] = PrometheusIntegration(**kwargs)
    
    def add_elasticsearch(self, **kwargs) -> None:
        """Add Elasticsearch integration."""
        self.integrations['elasticsearch'] = ElasticsearchIntegration(**kwargs)
    
    def add_redis(self, **kwargs) -> None:
        """Add Redis integration."""
        self.integrations['redis'] = RedisIntegration(**kwargs)
    
    def add_slack(self, **kwargs) -> None:
        """Add Slack integration."""
        self.integrations['slack'] = SlackIntegration(**kwargs)
    
    def add_discord(self, **kwargs) -> None:
        """Add Discord integration."""
        self.integrations['discord'] = DiscordIntegration(**kwargs)
    
    def log_metrics(self, metrics: Dict[str, Any], **kwargs) -> None:
        """Log metrics to all available integrations."""
        with self._lock:
            # Prometheus
            if 'prometheus' in self.integrations:
                prometheus = self.integrations['prometheus']
                for name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        prometheus.log_metric(name, float(value))
            
            # Redis caching
            if 'redis' in self.integrations and 'experiment_id' in kwargs:
                redis_int = self.integrations['redis']
                redis_int.cache_metrics(kwargs['experiment_id'], metrics)
    
    def log_entry(self, log_data: Dict[str, Any]) -> None:
        """Send log entry to all available integrations."""
        with self._lock:
            # Elasticsearch
            if 'elasticsearch' in self.integrations:
                elasticsearch = self.integrations['elasticsearch']
                elasticsearch.log_entry(log_data)
    
    def send_notification(
        self,
        message: str,
        level: str = "info",
        services: Optional[List[str]] = None
    ) -> None:
        """Send notification to specified services."""
        services = services or ['slack', 'discord']
        
        with self._lock:
            for service in services:
                if service in self.integrations:
                    integration = self.integrations[service]
                    integration.send_notification(message, level=level)
    
    def publish_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Publish event to available pub/sub systems."""
        with self._lock:
            if 'redis' in self.integrations:
                redis_int = self.integrations['redis']
                redis_int.publish_event(event_type, event_data)
    
    def get_available_integrations(self) -> List[str]:
        """Get list of available integrations."""
        available = []
        for name, integration in self.integrations.items():
            if hasattr(integration, 'available') and integration.available:
                available.append(name)
        return available


@contextmanager
def integration_context(**integrations_config):
    """
    Context manager for integrations.
    
    Args:
        **integrations_config: Configuration for various integrations
    
    Yields:
        IntegrationsManager instance
    """
    manager = IntegrationsManager()
    
    # Setup integrations based on config
    for integration_type, config in integrations_config.items():
        if integration_type == 'prometheus' and config:
            manager.add_prometheus(**config)
        elif integration_type == 'elasticsearch' and config:
            manager.add_elasticsearch(**config)
        elif integration_type == 'redis' and config:
            manager.add_redis(**config)
        elif integration_type == 'slack' and config:
            manager.add_slack(**config)
        elif integration_type == 'discord' and config:
            manager.add_discord(**config)
    
    try:
        yield manager
    finally:
        # Cleanup if needed
        pass


# Environment-based configuration helpers
def get_prometheus_config() -> Optional[Dict[str, Any]]:
    """Get Prometheus configuration from environment."""
    gateway_url = os.getenv('PROMETHEUS_GATEWAY_URL')
    if gateway_url:
        return {
            'gateway_url': gateway_url,
            'job_name': os.getenv('PROMETHEUS_JOB_NAME', 'refunc_ml')
        }
    return None


def get_elasticsearch_config() -> Optional[Dict[str, Any]]:
    """Get Elasticsearch configuration from environment."""
    hosts = os.getenv('ELASTICSEARCH_HOSTS')
    if hosts:
        return {
            'hosts': hosts.split(','),
            'index_name': os.getenv('ELASTICSEARCH_INDEX', 'ml-logs')
        }
    return None


def get_redis_config() -> Optional[Dict[str, Any]]:
    """Get Redis configuration from environment."""
    host = os.getenv('REDIS_HOST')
    if host:
        return {
            'host': host,
            'port': int(os.getenv('REDIS_PORT', '6379')),
            'db': int(os.getenv('REDIS_DB', '0')),
            'password': os.getenv('REDIS_PASSWORD'),
            'prefix': os.getenv('REDIS_PREFIX', 'refunc:ml:')
        }
    return None


def get_slack_config() -> Optional[Dict[str, Any]]:
    """Get Slack configuration from environment."""
    webhook_url = os.getenv('SLACK_WEBHOOK_URL')
    token = os.getenv('SLACK_TOKEN')
    
    if webhook_url or token:
        return {
            'webhook_url': webhook_url,
            'token': token
        }
    return None


def get_discord_config() -> Optional[Dict[str, Any]]:
    """Get Discord configuration from environment."""
    webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
    if webhook_url:
        return {'webhook_url': webhook_url}
    return None


def auto_configure_integrations() -> IntegrationsManager:
    """Auto-configure integrations from environment variables."""
    manager = IntegrationsManager()
    
    # Auto-configure based on environment
    prometheus_config = get_prometheus_config()
    if prometheus_config:
        manager.add_prometheus(**prometheus_config)
    
    elasticsearch_config = get_elasticsearch_config()
    if elasticsearch_config:
        manager.add_elasticsearch(**elasticsearch_config)
    
    redis_config = get_redis_config()
    if redis_config:
        manager.add_redis(**redis_config)
    
    slack_config = get_slack_config()
    if slack_config:
        manager.add_slack(**slack_config)
    
    discord_config = get_discord_config()
    if discord_config:
        manager.add_discord(**discord_config)
    
    return manager