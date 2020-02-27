import os
from abc import ABC, abstractmethod
from glob import glob
from urllib.parse import urlparse

import boto3
from pyspark.context import SparkContext

import settings


class AbstractLoader(ABC):
    """Class-skeleton for Loader-classes."""

    def __init__(self, root_path: str = '', *args, **kwargs):
        self.root_path = root_path
        super(AbstractLoader, self).__init__()

    @abstractmethod
    def get_path(self, *paths):
        """Get one full path from list of paths."""
        pass

    @abstractmethod
    def list_dir(self, raw_path: str, file_pattern: str):
        """List files, that satisfy `file_pattern`, in `raw_path` directory."""
        pass

    @abstractmethod
    def join(self, *paths):
        """Join paths with appropriate separator."""
        pass


class S3Loader(AbstractLoader):
    """Loader to work with s3 filesystem."""

    def __init__(self, sc: SparkContext, root_path: str = '', *args, **kwargs):

        super(S3Loader, self).__init__(root_path)

        self.access_id = settings.AWS_ACCESS_KEY_ID
        self.access_key = settings.AWS_SECRET_ACCESS_KEY
        self.bucket = self.__s3_bucket()
        self.root_path = urlparse(root_path, allow_fragments=False).path[1:]
        self.__set_hadoop_conf(sc)

    def __set_hadoop_conf(self, sc: SparkContext):
        """Update hadoop setting to work with S3."""

        hadoop_conf = sc._jsc.hadoopConfiguration()

        hadoop_conf.set("fs.s3n.impl",
                        "org.apache.hadoop.fs.s3native.NativeS3FileSystem")
        hadoop_conf.set("fs.s3n.awsAccessKeyId", self.access_id)
        hadoop_conf.set("fs.s3n.awsSecretAccessKey", self.access_key)

    def __s3_bucket(self):
        """Extract bucket name from `root_path`."""

        bucket_name = urlparse(self.root_path, allow_fragments=False).netloc
        session = boto3.Session(
            aws_access_key_id=self.access_id,
            aws_secret_access_key=self.access_key,
        )
        s3 = session.resource('s3')
        return s3.Bucket(bucket_name)

    def __get_path(self, path):
        return '/'.join(['s3n:/', self.bucket.name, path]).rstrip('/')

    def get_path(self, *paths):
        clean_paths = [p.rstrip('/') for p in paths if p is not None]
        r = '/'.join([self.__get_path(self.root_path), *clean_paths])
        return r

    def list_dir(self, raw_path, file_pattern: str):
        path = urlparse(self.get_path(raw_path, file_pattern), allow_fragments=False).path[1:]
        objects = [o.key for o in self.bucket.objects.filter(Prefix=path)]
        # TODO: don't return raw_path(rewrite better)
        objects = [self.__get_path(path) for path in objects]
        try:
            o = self.get_path(raw_path).rstrip('/') + '/'
            objects.remove(o)
        except ValueError as e:
            print(e)
        return objects

    def join(self, *paths):
        return '/'.join([path.rstrip('/') for path in paths])


class LocalStorageLoader(AbstractLoader):
    """Loader to work with local filesystem."""

    def get_path(self, *paths):
        return os.path.join(self.root_path, *paths)

    def list_dir(self, raw_path, file_pattern: str):
        path = self.get_path(raw_path, file_pattern + '*')
        return glob(path)

    def join(self, *paths):
        return os.path.join(*paths)


class LoaderStrategy:
    mapping = {
        's3': S3Loader,
        'local': LocalStorageLoader
    }

    @classmethod
    def get_loader(cls, storage_type: str = 'local'):
        try:
            return cls.mapping[storage_type]
        except KeyError as e:
            raise KeyError(
                f'Can\'t resolve loader for `storage_type`="{storage_type}"')
