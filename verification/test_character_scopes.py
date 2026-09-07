"""角色命名空间兼容性验证：内存 Qdrant，不读取或修改生产记忆。"""
import base64
import hashlib
import hmac
import json
import unittest
from urllib.parse import quote
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from memory_rag.api.app import create_app
from memory_rag.api.auth import ScopedMemoryAuth
from memory_rag.application.memory_service import MemoryService
from memory_rag.config import QdrantSettings
from memory_rag.infrastructure.qdrant_memory_repository import QdrantMemoryRepository

KEY="role-test-signing-"*4
def token(owner):
    payload=base64.urlsafe_b64encode(json.dumps(["memory-rag",owner,2],separators=(",",":")).encode()).decode().rstrip("=")
    unsigned="wac1."+payload
    signature=base64.urlsafe_b64encode(hmac.new(KEY.encode(),unsigned.encode(),hashlib.sha256).digest()).decode().rstrip("=")
    return unsigned+"."+signature

class Embedding:
    def embed(self,text):return [1.0,0.0,0.0]
    def close(self):pass

class CharacterMemoryTests(unittest.TestCase):
    def test_namespaced_vector_crud_is_authenticated_and_isolated(self):
        a="@club-role:"+hashlib.sha256(b"alice\0role-a").hexdigest()
        b="@club-role:"+hashlib.sha256(b"alice\0role-b").hexdigest()
        client=QdrantClient(":memory:")
        repository=QdrantMemoryRepository(QdrantSettings(url="http://unused.local",collection_name="role_test",vector_name="semantic",vector_size=3),client=client,ensure_payload_indexes=False)
        app=create_app(MemoryService(Embedding(),repository));app.add_middleware(ScopedMemoryAuth,signing_key=KEY)
        headers={"Authorization":"Bearer "+token(a)}
        root="/users/"+quote(a,safe="")+"/environments/2/memories"
        other="/users/"+quote(b,safe="")+"/environments/2/memories"
        body={"kind":"preference","title":"测试偏好","content":"测试角色A喜欢牛油果","tags":["test"],"importance":4,"confidence":0.8,"source":{"type":"conversation","ref":"test-role"},"fact_key":"test.avocado"}
        try:
            with TestClient(app) as http:
                created=http.post(root,json=body,headers=headers)
                self.assertEqual(201,created.status_code,created.text)
                memory_id=created.json()["id"]
                self.assertEqual(200,http.get(root+"/"+memory_id,headers=headers).status_code)
                self.assertEqual(403,http.get(other+"/"+memory_id,headers=headers).status_code)
                other_headers={"Authorization":"Bearer "+token(b)}
                self.assertEqual(404,http.get(other+"/"+memory_id,headers=other_headers).status_code)
                self.assertEqual([],http.post(other+"/search",json={"query":"牛油果"},headers=other_headers).json())
        finally:
            repository.close();client.close()

