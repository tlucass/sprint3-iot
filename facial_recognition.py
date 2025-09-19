import cv2
import numpy as np
import os
import pickle
import time
from datetime import datetime

class FacialAuthSystem:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                raise Exception("Nao foi possivel carregar o classificador Haar Cascade")
        
        self.face_data = []
        self.labels = []
        self.current_id = 0
        self.label_ids = {}
        self.trained = False
        
        self.security_level = "Medio"
        self.security_modes = {
            "Baixo": {"scale": 1.2, "neighbors": 3, "min_size": (30, 30), "confidence_threshold": 60},
            "Medio": {"scale": 1.1, "neighbors": 5, "min_size": (40, 40), "confidence_threshold": 70},
            "Alto": {"scale": 1.05, "neighbors": 7, "min_size": (50, 50), "confidence_threshold": 80}
        }
        
        self.access_history = []
        self.authenticated_user = None
        self.authentication_time = None
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Nao foi possivel acessar a camera")

        if not os.path.exists('user_data'):
            os.makedirs('user_data')
        if not os.path.exists('security_logs'):
            os.makedirs('security_logs')
            
        self.load_user_data()
    
    def set_security_level(self, level):
        """Define o nivel de seguranca"""
        if level in self.security_modes:
            config = self.security_modes[level]
            self.scale_factor = config["scale"]
            self.min_neighbors = config["neighbors"]
            self.min_size = config["min_size"]
            self.confidence_threshold = config["confidence_threshold"]
            self.security_level = level
            print(f"Nivel de seguranca alterado para: {level}")
            return True
        return False
    
    def detect_faces(self, frame):
        """Detecta rostos no frame usando Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )
        
        return faces, gray
    
    def draw_auth_interface(self, frame, faces, confidence=0, name="Nao identificado", status="Analisando..."):
        """Desenha interface de autenticacao"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (500, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        title = "Fiap Invest+ - Autenticacao Facial"
        cv2.putText(frame, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        
        status_color = (0, 255, 0) if "Concedido" in status else (0, 0, 255) if "Negado" in status else (200, 200, 200)
        cv2.putText(frame, f"Status: {status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # REMOVIDO: Contador de tentativas
        cv2.putText(frame, f"Seguranca: {self.security_level}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        for (x, y, w, h) in faces:
            color = (0, 255, 0) if "Concedido" in status else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            label = f"{name} ({confidence}%)" if confidence > 0 else "Verificando..."
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # REMOVIDO: Mensagem de bloqueio
        cv2.putText(frame, "Posicione seu rosto na camera | Q: Sair | R: Registrar", 
                   (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def register_user(self, user_name, user_id, sample_count=25):
        """Registra um novo usuario no sistema"""
        if user_id in [u["id"] for u in self.label_ids.values()]:
            print(f"ID de usuario {user_id} ja existe!")
            return False
            
        self.label_ids[self.current_id] = {"name": user_name, "id": user_id}
        
        print(f"Registrando {user_name} (ID: {user_id}). Coletando {sample_count} amostras...")
        
        count = 0
        samples = []
        while count < sample_count:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            faces, gray = self.detect_faces(frame)
            
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_resized = cv2.resize(roi_gray, (100, 100))
                
                samples.append(roi_resized)
                count += 1
                
                cv2.putText(frame, f"Registrando: {count}/{sample_count}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            frame = self.draw_auth_interface(frame, faces, 0, user_name, "Registrando...")
            cv2.imshow('Fiap Invest+ - Registro Facial', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if count >= 15:  
            self.face_data.extend(samples)
            self.labels.extend([self.current_id] * count)
            self.current_id += 1
            self.save_user_data()
            print(f"Usuario {user_name} registrado com sucesso!")
            cv2.destroyWindow('Fiap Invest+ - Registro Facial')
            return True
        else:
            print("Registro cancelado. Amostras insuficientes.")
            cv2.destroyWindow('Fiap Invest+ - Registro Facial')
            return False
    
    def authenticate_user(self, frame):
        """Autentica o usuario com reconhecimento facial"""
        if not self.trained or len(self.face_data) == 0:
            return None, 0, "Sistema nao treinado", []
        
        faces, gray = self.detect_faces(frame)
        
        if len(faces) == 0:
            return None, 0, "Nenhum rosto detectado", faces
        
        if len(faces) > 1:
            return None, 0, "Multiplos rostos detectados", faces
        
        x, y, w, h = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_gray, (100, 100))
        
        best_match_id = -1
        best_match_score = float('inf')
        
        for i, trained_face in enumerate(self.face_data):
            diff = cv2.absdiff(roi_resized, trained_face)
            score = np.mean(diff)
            
            if score < best_match_score:
                best_match_score = score
                best_match_id = self.labels[i]
        
        confidence = max(0, 100 - best_match_score)
        
        if confidence >= self.confidence_threshold and best_match_id in self.label_ids:
            user_info = self.label_ids[best_match_id]
            return user_info, confidence, "Acesso Concedido", faces
        else:
            return None, confidence, "Acesso Negado", faces
    
    def log_access_attempt(self, success, user_info=None, confidence=0):
        """Registra tentativa de acesso (sem bloqueio)"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "success": success,
            "user": user_info["name"] if user_info else "Unknown",
            "user_id": user_info["id"] if user_info else "N/A",
            "confidence": confidence,
            "security_level": self.security_level
        }
        
        self.access_history.append(log_entry)
        
        with open(f'security_logs/access_log_{datetime.now().strftime("%Y%m%d")}.txt', 'a') as f:
            status = "SUCESSO" if success else "FALHA"
            f.write(f"{timestamp} | {status} | {log_entry['user']} | {confidence}% | {self.security_level}\n")
    
    def save_user_data(self):
        """Salva dados dos usuarios"""
        user_data = {
            'face_data': self.face_data,
            'labels': self.labels,
            'label_ids': self.label_ids,
            'current_id': self.current_id
        }
        
        with open('user_data/user_profiles.pickle', 'wb') as f:
            pickle.dump(user_data, f)
        
        self.trained = len(self.face_data) > 0
        print("Dados de usuarios salvos!")
    
    def load_user_data(self):
        """Carrega dados dos usuarios"""
        try:
            with open('user_data/user_profiles.pickle', 'rb') as f:
                user_data = pickle.load(f)
            
            self.face_data = user_data['face_data']
            self.labels = user_data['labels']
            self.label_ids = user_data['label_ids']
            self.current_id = user_data['current_id']
            self.trained = len(self.face_data) > 0
            
            print(f"Dados carregados: {len(set(self.labels))} usuarios registrados")
            return True
        except:
            print("Nenhum dado de usuario encontrado. Iniciando sistema novo.")
            return False
    
    def show_success_countdown(self, frame, user_info, confidence, faces, duration=5):
        """Mostra contagem regressiva de sucesso antes de fechar"""
        start_time = time.time()
        
        while time.time() - start_time < duration:
            remaining = int(duration - (time.time() - start_time))
            
            frame_copy = frame.copy()
            frame_copy = self.draw_auth_interface(
                frame_copy, faces, confidence, user_info["name"], 
                f"Acesso Concedido! Fechando em {remaining}s..."
            )
            
            cv2.imshow('Fiap Invest+ - Autenticacao Facial', frame_copy)
            
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
    
    def run_authentication(self):
        """Executa o loop principal de autenticacao"""
        print("=== Fiap Invest+ - Sistema de Autenticacao Facial ===")
        print("Niveis de seguranca: 1-Baixo, 2-Medio, 3-Alto")
        print("Comandos: R-Registrar, Q-Sair, 1-3-Alterar seguranca")
        
        self.set_security_level("Medio")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Erro ao acessar camera")
                break
            
            user_info = None
            confidence = 0
            status = "Analisando..."
            faces = []
            
            faces, gray = self.detect_faces(frame)
            
            if len(faces) == 1:
                user_info, confidence, status, _ = self.authenticate_user(frame)
                
                if user_info and "Concedido" in status:
                    self.log_access_attempt(True, user_info, confidence)
                    self.authenticated_user = user_info
                    self.authentication_time = datetime.now()
                    
                    self.show_success_countdown(frame, user_info, confidence, faces, 5)
                    
                    print(f"Acesso concedido para {user_info['name']} ({confidence:.1f}%)")
                    self.show_app_welcome(user_info)
                    break

            user_name = user_info["name"] if user_info else "Nao identificado"
            frame = self.draw_auth_interface(frame, faces, confidence, user_name, status)
            cv2.imshow('Fiap Invest+ - Autenticacao Facial', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                success = self.register_new_user()
                if success:
                    print("Retornando a autenticacao...")
            elif key == ord('1'):
                self.set_security_level("Baixo")
            elif key == ord('2'):
                self.set_security_level("Medio")
            elif key == ord('3'):
                self.set_security_level("Alto")
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def register_new_user(self):
        """Interface para registro de novo usuario"""
        print("\n--- Registro de Novo Usuario ---")
        
        while True:
            user_name = input("Nome do usuario: ")
            if user_name.strip():
                break
            print("Nome nao pode estar vazio. Tente novamente.")
        
        while True:
            user_id = input("ID do usuario: ")
            if not user_id.strip():
                print("ID nao pode estar vazio. Tente novamente.")
                continue
            
            if user_id in [u["id"] for u in self.label_ids.values()]:
                print(f"ID '{user_id}' ja existe! Por favor, escolha outro ID.")
                print("IDs existentes:", [u["id"] for u in self.label_ids.values()])
                continue
            break
        
        success = self.register_user(user_name, user_id)
        if success:
            self.save_user_data()
            return True
        return False
    
    def show_app_welcome(self, user_info):
        """Simula a tela de boas-vindas do app de investimentos"""
        print("\n" + "="*50)
        print(f"BEM-VINDO AO Fiap Invest+, {user_info['name']}!")
        print("="*50)
        print(f"Autenticacao por reconhecimento facial concluida")
        print(f"Hora de acesso: {self.authentication_time.strftime('%H:%M:%S')}")
        print(f"Nivel de seguranca: {self.security_level}")
        print("\nSeu portfolio esta carregando...")
        print("\nRecursos disponiveis:")
        print("- Visualizacao de carteira de investimentos")
        print("- Analise de desempenho de ativos")
        print("- Recomendacoes personalizadas")
        print("- Simulador de investimentos")
        print("\nAcesso seguro concedido!")
        print("="*50)

def main():
    auth_system = FacialAuthSystem()
    auth_system.run_authentication()

if __name__ == "__main__":
    main()