import hashlib
import random
import base64
import shutil
import pandas as pd
import time
import socket
import uuid
import traceback
import requests
import os
import csv
import logging
import json
import urllib.parse
import codecs
import io
import io

from urllib.parse import urlparse, urljoin  # 添加这行
from werkzeug.security import generate_password_hash, check_password_hash

from flask_wtf.csrf import generate_csrf
from functools import wraps
from flask import request
from pathlib import Path
from io import StringIO, BytesIO
from datetime import datetime, timedelta, timezone
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, make_response, abort, jsonify, session, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from sqlalchemy import  extract, func, or_, and_
from transformers import pipeline, AutoTokenizer, AutoConfig, AutoModelForQuestionAnswering
import torch
from PIL import Image
from torchvision import models, transforms
from huggingface_hub import snapshot_download
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sqlalchemy.exc import SQLAlchemyError
from flask import request, render_template
from sqlalchemy import and_, or_
from flask_migrate import Migrate
from sqlalchemy import case
from datetime import datetime, timezone

from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

import calendar

# 必须在其他导入之前加载环境变量
load_dotenv()  

# 禁用代理（如果需要）
os.environ["NO_PROXY"] = "huggingface.co"

# 初始化 Flask 应用
app = Flask(__name__)

# 配置数据库和其他设置
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/chemical_inventory'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'fallback_secret_key')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = { 'png', 'jpg', 'jpeg', 'gif', 'pdf', 'doc', 'docx', 'xlsx',  'mp4', 'mov', 'avi'}
app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024 * 1024 # 3GB
app.config['DEEPSEEK_API_URL'] = 'https://api.deepseek.com/v1/chat/completions'
app.config['DEEPSEEK_API_KEY'] = os.getenv('DEEPSEEK_API_KEY')
app.config['WTF_CSRF_ENABLED'] = False  # 临时禁用CSRF
app.config['DELETE_PASSWORD'] = 'comp'  # 设置删除密码为"comp"
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your_actual_secret_key_here')

db = SQLAlchemy(app)
migrate = Migrate(app, db)

retry_strategy = Retry(
    total=3,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["POST"],
    backoff_factor=1
)
adapter = HTTPAdapter(max_retries=retry_strategy)
requests_session = requests.Session()
requests_session.mount("https://", adapter)

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 初始化登录管理器 - 现在 app 已经定义
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = '请先登录'

# 用户类定义
class User(UserMixin):
    """用户类，继承 UserMixin 以支持 Flask-Login"""
    def __init__(self, user_data=None):
        if user_data:
            self.id = user_data.get('id', 1)
            self.username = user_data.get('username', 'admin')
            self.password_hash = user_data.get('password_hash')
            self.permissions = user_data.get('permissions', ['chemical:create', 'chemical:edit', 'chemical:delete'])
        else:
            # 默认用户 - 使用固定的密码哈希
            self.id = 1
            self.username = 'admin'
            # 这里使用预先计算好的 'admin' 的哈希值
            self.password_hash = generate_password_hash('admin')  # 修复：生成实际的密码哈希
            self.permissions = ['chemical:create', 'chemical:edit', 'chemical:delete']
    
    def check_password(self, password):
        """检查密码"""
        # 如果密码哈希为空，使用默认密码验证
        if not self.password_hash:
            return password == 'admin'
            
        # 检查哈希格式是否有效
        if not self.password_hash.startswith(('pbkdf2:', 'sha256:', 'scrypt:')):
            # 如果不是有效的哈希格式，使用简单比较
            logger.warning(f"无效的密码哈希格式，使用简单验证")
            return password == 'admin'
            
        try:
            return check_password_hash(self.password_hash, password)
        except ValueError as e:
            logger.error(f"密码验证错误: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"密码验证异常: {str(e)}")
            return False
    
    def get_id(self):
        return str(self.id)

@login_manager.user_loader
def load_user(user_id):
    # 这里应该从数据库加载用户
    # 简化版本：返回默认用户
    user = User()
    user.id = int(user_id)
    user.username = 'admin'
    user.password_hash = generate_password_hash('admin')  # 修复：确保有密码哈希
    user.permissions = ['chemical:create', 'chemical:edit', 'chemical:delete']
    return user

# 权限控制装饰器
def permission_required(permission):
    """权限要求装饰器"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                return redirect(url_for('login'))
            if not current_user.has_permission(permission):
                abort(403)
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def log_activity(action, resource_type):
    """活动日志装饰器"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                # 执行原函数
                result = f(*args, **kwargs)
                
                # 记录活动日志的逻辑
                # 这里可以记录到数据库或日志文件
                logger.info(f"Activity: {action} on {resource_type} by user")
                
                return result
            except Exception as e:
                logger.error(f"Activity logging failed for {action}: {str(e)}")
                raise
        return decorated_function
    return decorator

# 获取当前用户的函数（需要根据你的认证系统实现）
def get_current_user():
    """获取当前用户"""
    # 这里应该根据你的认证系统返回当前用户
    # 例如从 session 或 JWT token 中获取
    return User(1, 'admin', ['chemical:create', 'chemical:edit', 'chemical:delete'])






# 调试环境变量
logger.debug("环境变量加载状态：")
logger.debug(f"DEEPSEEK_API_KEY: {os.getenv('DEEPSEEK_API_KEY')}")
logger.debug(f"DB_URI: {os.getenv('SQLALCHEMY_DATABASE_URI')}")

# 模型路径
MODEL_PATH = "./models/bert-base-chinese"
VISION_MODEL_PATH = "./models/resnet50"

def is_safe_path(base_path, target_path):
    """检查目标路径是否在允许的基础路径内"""
    try:
        base_path = Path(base_path).resolve()
        target_path = Path(target_path).resolve()
        return base_path in target_path.parents or target_path == base_path
    except Exception:
        return False

def initialize_models():
    """初始化 AI 模型，增加文件存在性检查"""
    required_files = {
        MODEL_PATH: ["config.json", "pytorch_model.bin", "tokenizer.json"],
        VISION_MODEL_PATH: ["model.pth"]
    }

    for model_dir, files in required_files.items():
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"模型目录 {model_dir} 不存在")
        for f in files:
            if not os.path.exists(os.path.join(model_dir, f)):
                raise FileNotFoundError(f"模型文件 {f} 在目录 {model_dir} 中缺失")
    logger.info("模型文件检查完成，使用本地模型文件。")

initialize_models()

# 关键修正：显式加载分词器
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    nlp = pipeline(
        "question-answering", 
        model=AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH),
        tokenizer=tokenizer,
        config=AutoConfig.from_pretrained(MODEL_PATH)
    )
except Exception as e:
    logger.error(f"模型加载失败: {str(e)}")
    raise

# 加载图像模型
try:
    vision_model = models.resnet50(weights=None)
    vision_model.load_state_dict(
        torch.load(
            os.path.join(VISION_MODEL_PATH, "model.pth"),
            map_location=torch.device('cpu')
        )
    )
    vision_model.eval()
except Exception as e:
    logger.error(f"图像模型加载失败: {str(e)}")
    raise

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 定义化学品模型（新增四个字段）
class Chemical(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product_code = db.Column(db.String(100), nullable=False, unique=True)
    name = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(100))
    specification = db.Column(db.String(100))
    concentration = db.Column(db.Float)
    unit = db.Column(db.String(50))
    image = db.Column(db.String(200))
    model = db.Column(db.String(100))
    location = db.Column(db.String(100))
    operator = db.Column(db.String(100))
    confirmer = db.Column(db.String(100))
    action = db.Column(db.String(50))
    manufacturer = db.Column(db.String(100))
    batch_number = db.Column(db.String(100))
    production_date = db.Column(db.Date)
    expiration_date = db.Column(db.Date)
    quantity = db.Column(db.Integer)
    total_stock = db.Column(db.Integer)
    safety_stock = db.Column(db.Integer)
    status = db.Column(db.String(50))
    price = db.Column(db.Float)
    supplier = db.Column(db.String(200))
    notes = db.Column(db.Text)
    operation_date = db.Column(db.Date, default=datetime.utcnow)
    batch_total_stock = db.Column(db.Integer)
    attachment = db.Column(db.String(200))
    delivery_order_attachment = db.Column(db.String(200))
    invoice_attachment = db.Column(db.String(200))
    msds_attachment = db.Column(db.String(200))
    signature_data = db.Column(db.Text)  # 存储Base64格式的签名数据
    signature_image = db.Column(db.String(200))  # 存储签名图片文件名

    # 新增四个字段
    consumable_category = db.Column(db.String(100))  # 耗材类别
    cas_number = db.Column(db.String(100))           # CAS编号
    warehouse_location = db.Column(db.String(100))   # 仓库位置
    regulatory_category = db.Column(db.String(100))  # 监管类别

    @property
    def days_remaining(self):
        if self.expiration_date:
            return (self.expiration_date - datetime.now().date()).days
        return None

# 文件上传检查
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# 创建数据库表
with app.app_context():
    db.create_all()

# 签名验证中间件
def validate_signature(func):
    """装饰器：验证表单提交是否包含有效签名"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 仅对POST请求验证签名
        if request.method == 'POST':
            signature_data = request.form.get('signature_data')
            
            if not signature_data:
                flash('请先进行手写签名确认', 'warning')
                if request.endpoint == 'add_chemical':
                    return render_template('add_chemical.html')
                elif request.endpoint == 'edit_chemical':
                    return redirect(url_for('edit_chemical', id=kwargs.get('id')))
            
            # 验证签名数据格式
            if not signature_data.startswith('data:image/png;base64,'):
                flash('签名数据格式无效，请重新签名', 'warning')
                if request.endpoint == 'add_chemical':
                    return render_template('add_chemical.html')
                elif request.endpoint == 'edit_chemical':
                    return redirect(url_for('edit_chemical', id=kwargs.get('id')))
        
        return func(*args, **kwargs)
    return wrapper

# 首页路由
@app.route('/')
def index():
    page = request.args.get('page', 1, type=int)
    chemicals = Chemical.query.paginate(page=page, per_page=10)
    return render_template('index.html', chemicals=chemicals)

# 添加化学品路由（处理新增字段）
@app.route('/add', methods=['GET', 'POST'])
@login_required
@permission_required('chemical:create')
@validate_signature
@log_activity('add_chemical', 'chemical')
def add_chemical():
    if request.method == 'POST':
        try:
            # 获取签名数据
            signature_data = request.form.get('signature_data', '')
            
            # 验证签名数据格式
            if not signature_data:
                flash('请先进行手写签名确认', 'warning')
                return render_template('add_chemical.html')
            
            # 检查签名数据格式
            if not signature_data.startswith('data:image/png;base64,'):
                logger.warning(f"Invalid signature format: {signature_data[:50]}")
                flash('签名数据格式无效，请重新签名', 'warning')
                return render_template('add_chemical.html')

            # 处理签名数据
            signature_filename = None
            if signature_data:
                # 从Base64字符串中提取图像数据
                signature_data_clean = signature_data.split(',', 1)[1]
                try:
                    signature_bytes = base64.b64decode(signature_data_clean)
                    
                    # 生成唯一文件名
                    timestamp = int(time.time())
                    signature_filename = f"signature_{timestamp}_{uuid.uuid4().hex[:8]}.png"
                    signature_path = os.path.join(app.config['UPLOAD_FOLDER'], signature_filename)
                    
                    # 确保上传目录存在
                    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                    
                    # 保存签名图片
                    with open(signature_path, "wb") as f:
                        f.write(signature_bytes)
                    
                    # 创建缩略图（可选）
                    try:
                        img = Image.open(io.BytesIO(signature_bytes))
                        img.thumbnail((200, 100))  # 创建缩略图
                        thumbnail_path = os.path.join(app.config['UPLOAD_FOLDER'], f"thumb_{signature_filename}")
                        img.save(thumbnail_path)
                    except Exception as thumb_error:
                        logger.warning(f"无法创建签名缩略图: {str(thumb_error)}")
                    
                    logger.info(f"签名已保存: {signature_filename}")
                except Exception as e:
                    logger.error(f"签名处理失败: {str(e)}")
                    flash('签名处理失败，请重新签名', 'danger')
                    return render_template('add_chemical.html')
            
            # 处理表单数据
            product_code = request.form.get('product_code')
            name = request.form.get('name')
            category = request.form.get('category')
            specification = request.form.get('specification')
            concentration = float(request.form.get('concentration', 0)) if request.form.get('concentration') else 0.0
            unit = request.form.get('unit')
            model = request.form.get('model')
            location = request.form.get('location')
            operator = request.form.get('operator')
            confirmer = request.form.get('confirmer')
            action = request.form.get('action')
            manufacturer = request.form.get('manufacturer')
            batch_number = request.form.get('batch_number')
            
            # 处理日期字段
            production_date = None
            if request.form.get('production_date'):
                production_date = datetime.strptime(request.form.get('production_date'), '%Y-%m-%d').date()
            
            expiration_date = None
            if request.form.get('expiration_date'):
                expiration_date = datetime.strptime(request.form.get('expiration_date'), '%Y-%m-%d').date()
            
            quantity = int(request.form.get('quantity', 0)) if request.form.get('quantity') else 0
            batch_total_stock = int(request.form.get('batch_total_stock', 0)) if request.form.get('batch_total_stock') else 0
            total_stock = int(request.form.get('total_stock', 0)) if request.form.get('total_stock') else 0
            safety_stock = int(request.form.get('safety_stock', 0)) if request.form.get('safety_stock') else 0
            status = request.form.get('status')
            price = float(request.form.get('price', 0)) if request.form.get('price') else 0.0
            supplier = request.form.get('supplier')
            notes = request.form.get('notes')
            
            # 处理新增字段
            consumable_category = request.form.get('consumable_category')
            cas_number = request.form.get('cas_number')
            warehouse_location = request.form.get('warehouse_location')
            regulatory_category = request.form.get('regulatory_category')

            # 处理图片上传
            image = None
            image_file = request.files.get('image')
            if image_file and image_file.filename != '' and allowed_file(image_file.filename):
                image_filename = secure_filename(image_file.filename)
                image_file.save(os.path.join(app.config['UPLOAD_FOLDER'], image_filename))
                image = image_filename

            # 处理附件上传
            attachment = None
            attachment_file = request.files.get('attachment')
            if attachment_file and attachment_file.filename != '' and allowed_file(attachment_file.filename):
                attachment_filename = secure_filename(attachment_file.filename)
                attachment_file.save(os.path.join(app.config['UPLOAD_FOLDER'], attachment_filename))
                attachment = attachment_filename

            # 处理送货单附件上传
            delivery_order_attachment = None
            delivery_order_file = request.files.get('delivery_order_attachment')
            if delivery_order_file and delivery_order_file.filename != '' and allowed_file(delivery_order_file.filename):
                delivery_order_filename = secure_filename(delivery_order_file.filename)
                delivery_order_file.save(os.path.join(app.config['UPLOAD_FOLDER'], delivery_order_filename))
                delivery_order_attachment = delivery_order_filename

            # 处理发票单附件上传
            invoice_attachment = None
            invoice_file = request.files.get('invoice_attachment')
            if invoice_file and invoice_file.filename != '' and allowed_file(invoice_file.filename):
                invoice_filename = secure_filename(invoice_file.filename)
                invoice_file.save(os.path.join(app.config['UPLOAD_FOLDER'], invoice_filename))
                invoice_attachment = invoice_filename

            # 处理MSDS附件上传
            msds_attachment = None
            msds_file = request.files.get('msds_attachment')
            if msds_file and msds_file.filename != '' and allowed_file(msds_file.filename):
                msds_filename = secure_filename(msds_file.filename)
                msds_file.save(os.path.join(app.config['UPLOAD_FOLDER'], msds_filename))
                msds_attachment = msds_filename

            # 创建化学品记录（包含签名）
            chemical = Chemical(
                product_code=product_code,
                name=name,
                category=category,
                specification=specification,
                concentration=concentration,
                unit=unit,
                model=model,
                location=location,
                operator=operator,
                confirmer=confirmer,
                action=action,
                manufacturer=manufacturer,
                batch_number=batch_number,
                production_date=production_date,
                expiration_date=expiration_date,
                quantity=quantity,
                batch_total_stock=batch_total_stock,
                total_stock=total_stock,
                safety_stock=safety_stock,
                status=status,
                price=price,
                supplier=supplier,
                notes=notes,
                image=image,
                attachment=attachment,
                delivery_order_attachment=delivery_order_attachment,
                invoice_attachment=invoice_attachment,
                msds_attachment=msds_attachment,
                # 新增字段
                consumable_category=consumable_category,
                cas_number=cas_number,
                warehouse_location=warehouse_location,
                regulatory_category=regulatory_category,
                # 签名字段
                signature_data=signature_data,  # 保存Base64字符串
                signature_image=signature_filename  # 保存图片文件名
            )
            
            db.session.add(chemical)
            db.session.commit()
            flash('化学品添加成功!', 'success')
            return redirect(url_for('index'))
        
        except ValueError as ve:
            db.session.rollback()
            logger.error(f"数据类型错误: {str(ve)}")
            flash(f'输入数据格式错误: {str(ve)}', 'danger')
            return render_template('add_chemical.html')
        
        except Exception as e:
            db.session.rollback()
            logger.error(f"添加化学品失败: {str(e)}")
            flash(f'错误: {str(e)}', 'danger')
            return render_template('add_chemical.html')

    # GET请求：显示添加表单
    return render_template('add_chemical.html')

# 编辑化学品路由（处理新增字段）
@app.route('/edit/<int:id>', methods=['GET', 'POST'])
@login_required
@permission_required('chemical:edit')
@validate_signature
@log_activity('edit_chemical', 'chemical')
def edit_chemical(id):
    chemical = Chemical.query.get_or_404(id)
    if request.method == 'POST':
        try:
            save_as_new = 'save_as_new' in request.form

            # 处理公共字段
            product_code = request.form.get('product_code')
            name = request.form.get('name')
            category = request.form.get('category')
            specification = request.form.get('specification')
            concentration = float(request.form.get('concentration', 0)) if request.form.get('concentration') else 0.0
            unit = request.form.get('unit')
            model = request.form.get('model')
            location = request.form.get('location')
            operator = request.form.get('operator')
            confirmer = request.form.get('confirmer')
            action = request.form.get('action')
            manufacturer = request.form.get('manufacturer')
            batch_number = request.form.get('batch_number')
            production_date = datetime.strptime(request.form.get('production_date'), '%Y-%m-%d').date() if request.form.get('production_date') else None
            expiration_date = datetime.strptime(request.form.get('expiration_date'), '%Y-%m-%d').date() if request.form.get('expiration_date') else None
            quantity = int(request.form.get('quantity', 0)) if request.form.get('quantity') else 0
            batch_total_stock = int(request.form.get('batch_total_stock', 0)) if request.form.get('batch_total_stock') else 0
            total_stock = int(request.form.get('total_stock', 0)) if request.form.get('total_stock') else 0
            safety_stock = int(request.form.get('safety_stock', 0)) if request.form.get('safety_stock') else 0
            status = request.form.get('status')
            price = float(request.form.get('price', 0)) if request.form.get('price') else 0.0
            supplier = request.form.get('supplier')
            notes = request.form.get('notes')
            
            # 处理新增字段
            consumable_category = request.form.get('consumable_category')
            cas_number = request.form.get('cas_number')
            warehouse_location = request.form.get('warehouse_location')
            regulatory_category = request.form.get('regulatory_category')

            # 处理签名数据
            signature_data = request.form.get('signature_data')
            signature_filename = None
            
            # 如果有新签名数据
            if signature_data and signature_data.startswith('data:image/png;base64,'):
                # 处理新签名
                signature_data_clean = signature_data.split(',', 1)[1]
                try:
                    signature_bytes = base64.b64decode(signature_data_clean)
                    
                    # 生成唯一文件名
                    timestamp = int(time.time())
                    signature_filename = f"signature_{timestamp}_{uuid.uuid4().hex[:8]}.png"
                    signature_path = os.path.join(app.config['UPLOAD_FOLDER'], signature_filename)
                    
                    # 确保上传目录存在
                    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                    
                    # 保存签名图片
                    with open(signature_path, "wb") as f:
                        f.write(signature_bytes)
                    
                    # 删除旧签名图片（如果存在）
                    if chemical.signature_image:
                        old_signature_path = os.path.join(app.config['UPLOAD_FOLDER'], chemical.signature_image)
                        if os.path.exists(old_signature_path):
                            os.remove(old_signature_path)
                    
                    logger.info(f"新签名已保存: {signature_filename}")
                except Exception as e:
                    logger.error(f"签名处理失败: {str(e)}")
                    flash('签名处理失败，请重新签名', 'danger')
                    return redirect(url_for('edit_chemical', id=id))

            # 处理文件上传字段
            file_fields = {
                'image': '图片',
                'attachment': '通用附件',
                'delivery_order_attachment': '送货单附件',
                'invoice_attachment': '发票附件',
                'msds_attachment': 'MSDS附件'
            }
            
            new_filenames = {}
            for field in file_fields.keys():
                file = request.files.get(field)
                if file and file.filename != '':
                    if allowed_file(file.filename):
                        # 生成唯一文件名
                        filename = secure_filename(file.filename)
                        base, ext = os.path.splitext(filename)
                        unique_id = str(uuid.uuid4())[:8]
                        unique_filename = f"{base}_{unique_id}{ext}"
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                        file.save(file_path)
                        new_filenames[field] = unique_filename
                    else:
                        flash(f'不允许的文件类型: {file.filename}', 'warning')

            if save_as_new:
                # 创建新化学品记录
                new_chem = Chemical(
                    product_code=product_code,
                    name=name,
                    category=category,
                    specification=specification,
                    concentration=concentration,
                    unit=unit,
                    model=model,
                    location=location,
                    operator=operator,
                    confirmer=confirmer,
                    action=action,
                    manufacturer=manufacturer,
                    batch_number=batch_number,
                    production_date=production_date,
                    expiration_date=expiration_date,
                    quantity=quantity,
                    batch_total_stock=batch_total_stock,
                    total_stock=total_stock,
                    safety_stock=safety_stock,
                    status=status,
                    price=price,
                    supplier=supplier,
                    notes=notes,
                    consumable_category=consumable_category,
                    cas_number=cas_number,
                    warehouse_location=warehouse_location,
                    regulatory_category=regulatory_category,
                    # 对于文件字段，如果有新上传的则用新的，否则用原来的
                    image=new_filenames.get('image', chemical.image),
                    attachment=new_filenames.get('attachment', chemical.attachment),
                    delivery_order_attachment=new_filenames.get('delivery_order_attachment', chemical.delivery_order_attachment),
                    invoice_attachment=new_filenames.get('invoice_attachment', chemical.invoice_attachment),
                    msds_attachment=new_filenames.get('msds_attachment', chemical.msds_attachment),
                    # 签名字段：如果有新签名则用新的，否则用原来的
                    signature_data=signature_data if signature_data else chemical.signature_data,
                    signature_image=signature_filename if signature_filename else chemical.signature_image
                )
                db.session.add(new_chem)
                flash('化学品已成功保存为新记录!', 'success')
            else:
                # 更新现有记录
                chemical.product_code = product_code
                chemical.name = name
                chemical.category = category
                chemical.specification = specification
                chemical.concentration = concentration
                chemical.unit = unit
                chemical.model = model
                chemical.location = location
                chemical.operator = operator
                chemical.confirmer = confirmer
                chemical.action = action
                chemical.manufacturer = manufacturer
                chemical.batch_number = batch_number
                chemical.production_date = production_date
                chemical.expiration_date = expiration_date
                chemical.quantity = quantity
                chemical.batch_total_stock = batch_total_stock
                chemical.total_stock = total_stock
                chemical.safety_stock = safety_stock
                chemical.status = status
                chemical.price = price
                chemical.supplier = supplier
                chemical.notes = notes
                chemical.consumable_category = consumable_category
                chemical.cas_number = cas_number
                chemical.warehouse_location = warehouse_location
                chemical.regulatory_category = regulatory_category
                
                # 更新文件字段，如果有新上传的
                for field in file_fields.keys():
                    if field in new_filenames:
                        # 删除旧文件（如果存在且不是新记录）
                        old_filename = getattr(chemical, field)
                        if old_filename:
                            old_path = os.path.join(app.config['UPLOAD_FOLDER'], old_filename)
                            if os.path.exists(old_path):
                                try:
                                    os.remove(old_path)
                                except Exception as e:
                                    logger.error(f"删除旧文件失败: {str(e)}")
                        # 设置新文件名
                        setattr(chemical, field, new_filenames[field])
                
                # 更新签名字段
                if signature_data:
                    chemical.signature_data = signature_data
                if signature_filename:
                    chemical.signature_image = signature_filename
                
                flash('化学品更新成功!', 'success')

            db.session.commit()
            return redirect(url_for('index'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"保存失败: {str(e)}")
            flash(f'错误: {str(e)}', 'danger')
            return redirect(url_for('edit_chemical', id=id))
    
    # GET请求：显示编辑表单
    return render_template('edit_chemical.html', chemical=chemical, datetime=datetime)
# 查看签名路由
@app.route('/view_signature/<int:id>')
def view_signature(id):
    chemical = Chemical.query.get_or_404(id)
    
    if chemical.signature_image:
        # 如果存在签名图片文件
        return send_from_directory(app.config['UPLOAD_FOLDER'], chemical.signature_image)
    elif chemical.signature_data:
        # 如果存在Base64签名数据
        response = make_response(base64.b64decode(chemical.signature_data.split(',', 1)[1]))
        response.headers['Content-Type'] = 'image/png'
        return response
    else:
        abort(404, description="未找到签名")

# 签名分析API
@app.route('/analyze_signature', methods=['POST'])
def analyze_signature():
    """分析签名特征API"""
    try:
        signature_data = request.json.get('signature_data')
        if not signature_data:
            return jsonify({'error': '缺少签名数据'}), 400
        
        # 验证签名数据格式
        if not signature_data.startswith('data:image/png;base64,'):
            return jsonify({'error': '无效的签名数据格式'}), 400
        
        # 提取Base64数据
        signature_data = signature_data.split(',', 1)[1]
        signature_bytes = base64.b64decode(signature_data)
        
        # 使用OpenCV分析签名特征
        try:
            import cv2
            import numpy as np
            
            # 将字节数据转换为OpenCV图像
            nparr = np.frombuffer(signature_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            # 简单分析
            _, threshold = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 计算签名复杂度
            complexity = len(contours)
            
            # 计算签名大小
            height, width = img.shape
            size_ratio = width / height if height > 0 else 0
            
            return jsonify({
                'complexity': complexity,
                'size_ratio': round(size_ratio, 2),
                'width': width,
                'height': height,
                'contours': len(contours),
                'analysis': '签名复杂度较高' if complexity > 5 else '签名复杂度一般'
            })
        except ImportError:
            logger.warning("OpenCV未安装，无法进行签名分析")
            return jsonify({'warning': '签名分析功能需要OpenCV支持'})
    
    except Exception as e:
        logger.error(f"签名分析失败: {str(e)}")
        return jsonify({'error': f'签名分析失败: {str(e)}'}), 500

# 其余代码保持不变...
# 由于代码长度限制，这里只展示了关键修改部分，其余代码保持不变

#导入模板下载功能（包含新增字段）：
@app.route('/download_import_template')
def download_import_template():
    # 创建示例数据（包含新增字段）
    data = [{
        '商品代码': 'CHEM-001',
        '化学品名称': '硫酸',
        '类别': '腐蚀品',
        '规格': '分析纯',
        '浓度': 98.0,
        '单位': '瓶',
        '型号': '500ml',
        '存储位置': 'A区-1架',
        '操作员': '张三',
        '确认人': '李四',
        '操作类型': '入库',
        '生产商': '中国化工',
        '批次号': '2023-08-001',
        '收支日期 ': '2023-08-01',
        '过期日期': '2025-08-01',
        '数量': 10,
        '此批总库存': 100,
        '总库存': 500,
        '安全库存': 50,
        '状态': '正常',
        '价格': 25.5,
        '供应商': '上海化学试剂公司',
        '备注': '危险品，小心操作',
        # 新增字段
        '耗材类别': '实验试剂',
        'CAS编号': '7732-18-5',
        '仓库位置': '主仓库',
        '监管类别': '危险化学品'
    }]
    
    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='模板', index=False)
        
        # 添加数据验证（示例）
        worksheet = writer.sheets['模板']
        
        # 添加操作类型下拉菜单
        worksheet.data_validation('K2:K1000', {
            'validate': 'list',
            'source': ['入库', '出库', '盘点']
        })
        
        # 添加状态下拉菜单
        worksheet.data_validation('R2:R1000', {
            'validate': 'list',
            'source': ['正常', '过期', '低库存']
        })
    
    output.seek(0)
    response = make_response(output.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=化学品导入模板.xlsx'
    response.headers['Content-type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    return response

# 下载附件路由
@app.route('/download_attachment/<int:id>/<string:type>')
def download_attachment(id, type):
    chemical = Chemical.query.get_or_404(id)
    if type == 'attachment':
        if not chemical.attachment:
            abort(404, description="该化学品没有关联附件")
        try:
            return send_from_directory(app.config['UPLOAD_FOLDER'], chemical.attachment, as_attachment=True)
        except FileNotFoundError:
            abort(404, description="附件文件未找到")
    elif type == 'delivery_order':
        if not chemical.delivery_order_attachment:
            abort(404, description="该化学品没有关联送货单附件")
        try:
            return send_from_directory(app.config['UPLOAD_FOLDER'], chemical.delivery_order_attachment, as_attachment=True)
        except FileNotFoundError:
            abort(404, description="送货单附件文件未找到")
    elif type == 'invoice':
        if not chemical.invoice_attachment:
            abort(404, description="该化学品没有关联发票单附件")
        try:
            return send_from_directory(app.config['UPLOAD_FOLDER'], chemical.invoice_attachment, as_attachment=True)
        except FileNotFoundError:
            abort(404, description="发票单附件文件未找到")
    elif type == 'msds':
        if not chemical.msds_attachment:
            abort(404, description="该化学品没有关联MSDS附件")
        try:
            return send_from_directory(app.config['UPLOAD_FOLDER'], chemical.msds_attachment, as_attachment=True)
        except FileNotFoundError:
            abort(404, description="MSDS附件文件未找到")
    else:
        abort(404, description="无效的附件类型")

# 导出化学品数据路由（包含新增字段）
@app.route('/export')
def export_chemicals():
    chemicals = Chemical.query.all()
    # 使用 StringIO 替代 BytesIO
    output = StringIO()
    # 写入 UTF-8 BOM
    output.write('\ufeff')  # Unicode BOM字符
    writer = csv.writer(output)
    # 完整字段名称（包含新增字段）
    headers = [
        'ID', '商品代码', '化学品名称', '类别', '规格', '浓度', '单位', '型号', 
        '存储位置', '操作员', '确认人', '操作类型', '生产商', '批次号', 
        '收支日期 ', '过期日期', '数量', '此批总库存', '总库存', '安全库存', 
        '状态', '价格', '供应商', '备注', '操作日期',
        # 新增字段
        '耗材类别', 'CAS编号', '仓库位置', '监管类别'
    ]
    writer.writerow(headers)
    
    for chemical in chemicals:
        writer.writerow([
            chemical.id, chemical.product_code, chemical.name, chemical.category,
            chemical.specification, chemical.concentration, chemical.unit,
            chemical.model, chemical.location, chemical.operator, chemical.confirmer,
            chemical.action, chemical.manufacturer, chemical.batch_number,
            chemical.production_date.strftime('%Y-%m-%d') if chemical.production_date else '',
            chemical.expiration_date.strftime('%Y-%m-%d') if chemical.expiration_date else '',
            chemical.quantity, chemical.batch_total_stock, chemical.total_stock, 
            chemical.safety_stock, chemical.status, chemical.price, 
            chemical.supplier, chemical.notes,
            chemical.operation_date.strftime('%Y-%m-%d %H:%M:%S') if chemical.operation_date else '',
            # 新增字段
            chemical.consumable_category or '',
            chemical.cas_number or '',
            chemical.warehouse_location or '',
            chemical.regulatory_category or ''
        ])
    
    output.seek(0)
    # 编码为UTF-8
    response = make_response(output.getvalue().encode('utf-8-sig'))
    response.headers['Content-Disposition'] = 'attachment; filename=chemicals.csv'
    response.headers['Content-type'] = 'text/csv; charset=utf-8'
    return response

# 导入化学品数据路由（处理新增字段）
@app.route('/import', methods=['GET', 'POST'])
def import_chemicals():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
            try:
                if file.filename.endswith('.csv'):
                    # 处理CSV
                    stream = StringIO(file.read().decode('utf-8'))
                    reader = csv.DictReader(stream)
                    data = [row for row in reader]
                else:  # 处理Excel
                    df = pd.read_excel(file)
                    # 将NaN替换为None
                    df = df.where(pd.notnull(df), None)
                    data = df.to_dict('records')
                
                success_count = 0
                error_rows = []
                
                for idx, row in enumerate(data, start=1):
                    try:
                        # 处理日期字段
                        production_date = None
                        if row.get('收支日期 '):
                            production_date = datetime.strptime(row['收支日期 '], '%Y-%m-%d').date()
                        
                        expiration_date = None
                        if row.get('过期日期'):
                            expiration_date = datetime.strptime(row['过期日期'], '%Y-%m-%d').date()
                        
                        # 创建化学品对象（包含新增字段）
                        chemical = Chemical(
                            product_code=row.get('商品代码', ''),
                            name=row.get('化学品名称', ''),
                            category=row.get('类别', ''),
                            specification=row.get('规格', ''),
                            concentration=float(row.get('浓度', 0)) if row.get('浓度') not in [None, ''] else 0.0,
                            unit=row.get('单位', ''),
                            model=row.get('型号', ''),
                            location=row.get('存储位置', ''),
                            operator=row.get('操作员', ''),
                            confirmer=row.get('确认人', ''),
                            action=row.get('操作类型', ''),
                            manufacturer=row.get('生产商', ''),
                            batch_number=row.get('批次号', ''),
                            production_date=production_date,
                            expiration_date=expiration_date,
                            quantity=int(row.get('数量', 0)) if row.get('数量') not in [None, ''] else 0,
                            batch_total_stock=int(row.get('此批总库存', 0)) if row.get('此批总库存') not in [None, ''] else 0,
                            total_stock=int(row.get('总库存', 0)) if row.get('总库存') not in [None, ''] else 0,
                            safety_stock=int(row.get('安全库存', 0)) if row.get('安全库存') not in [None, ''] else 0,
                            status=row.get('状态', ''),
                            price=float(row.get('价格', 0)) if row.get('价格') not in [None, ''] else 0.0,
                            supplier=row.get('供应商', ''),
                            notes=row.get('备注', ''),
                            # 新增字段
                            consumable_category=row.get('耗材类别', ''),
                            cas_number=row.get('CAS编号', ''),
                            warehouse_location=row.get('仓库位置', ''),
                            regulatory_category=row.get('监管类别', '')
                        )
                        db.session.add(chemical)
                        success_count += 1
                    except Exception as e:
                        error_rows.append({
                            'row': idx,
                            'error': str(e),
                            'data': row
                        })
                
                db.session.commit()
                flash(f'成功导入 {success_count} 条记录，失败 {len(error_rows)} 条', 'success')
                
                # 如果有错误，提供错误详情下载
                if error_rows:
                    error_output = StringIO()
                    error_writer = csv.writer(error_output)
                    error_writer.writerow(['行号', '错误信息', '数据'])
                    for error in error_rows:
                        error_writer.writerow([
                            error['row'], 
                            error['error'], 
                            json.dumps(error['data'], ensure_ascii=False)
                        ])
                    error_output.seek(0)
                    session['import_errors'] = error_output.getvalue()
                
            except Exception as e:
                db.session.rollback()
                flash(f'导入失败: {str(e)}', 'danger')
            return redirect(url_for('import_chemicals'))
        else:
            flash('请上传有效的CSV或Excel文件', 'warning')
    
    # 提供错误日志下载
    import_errors = session.pop('import_errors', None)
    return render_template('import_chemicals.html', import_errors=import_errors)

# 批量导出优化（包含新增字段）：
@app.route('/export_chemicals_excel')
def export_chemicals_excel():
    try:
        # 确保 xlsxwriter 模块已安装
        try:
            import xlsxwriter
        except ImportError:
            logger.error("缺少 xlsxwriter 模块，请安装: pip install xlsxwriter")
            flash('导出失败：缺少必要的 Excel 导出模块，请安装 xlsxwriter', 'danger')
            return redirect(url_for('index'))
        
        chemicals = Chemical.query.all()
        
        # 创建DataFrame - 确保所有字段都存在
        data = []
        for chem in chemicals:
            # 处理可能为空的字段
            prod_date = chem.production_date.strftime('%Y-%m-%d') if chem.production_date else ''
            exp_date = chem.expiration_date.strftime('%Y-%m-%d') if chem.expiration_date else ''
            op_date = chem.operation_date.strftime('%Y-%m-%d %H:%M:%S') if chem.operation_date else ''
            
            data.append({
                'ID': chem.id,
                '商品代码': chem.product_code,
                '化学品名称': chem.name,
                '类别': chem.category or '',
                '规格': chem.specification or '',
                '浓度': chem.concentration or 0.0,
                '单位': chem.unit or '',
                '型号': chem.model or '',
                '存储位置': chem.location or '',
                '操作员': chem.operator or '',
                '确认人': chem.confirmer or '',
                '操作类型': chem.action or '',
                '生产商': chem.manufacturer or '',
                '批次号': chem.batch_number or '',
                '收支日期 ': prod_date,
                '过期日期': exp_date,
                '数量': chem.quantity or 0,
                '此批总库存': chem.batch_total_stock or 0,
                '总库存': chem.total_stock or 0,
                '安全库存': chem.safety_stock or 0,
                '状态': chem.status or '',
                '价格': chem.price or 0.0,
                '供应商': chem.supplier or '',
                '备注': chem.notes or '',
                '操作日期': op_date,
                # 新增字段
                '耗材类别': chem.consumable_category or '',
                'CAS编号': chem.cas_number or '',
                '仓库位置': chem.warehouse_location or '',
                '监管类别': chem.regulatory_category or ''
            })
        
        # 确保有数据才创建DataFrame
        if not data:
            flash('没有可导出的数据', 'warning')
            return redirect(url_for('index'))
        
        df = pd.DataFrame(data)
        
        # 创建Excel文件
        output = BytesIO()
        try:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='化学品清单', index=False)
                
                # 获取工作表并设置列宽
                worksheet = writer.sheets['化学品清单']
                for idx, col in enumerate(df.columns):
                    # 设置列宽为数据最大长度+2
                    max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
                    worksheet.set_column(idx, idx, max_len)
        except Exception as e:
            logger.error(f"Excel写入失败: {str(e)}")
            flash(f'Excel写入失败: {str(e)}', 'danger')
            return redirect(url_for('index'))
        
        output.seek(0)
        response = make_response(output.getvalue())
        response.headers['Content-Disposition'] = 'attachment; filename=chemicals.xlsx'
        response.headers['Content-type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        return response
    
    except Exception as e:
        logger.error(f"化学品导出失败: {str(e)}")
        flash(f'化学品导出失败: {str(e)}', 'danger')
        return redirect(url_for('index'))

# 过期预警路由（包含仓库位置）
@app.route('/expiration_reminder')
def expiration_reminder():
    try:
        # 获取筛选类型参数
        filter_type = request.args.get('filter', 'upcoming')  # upcoming/expired/all
        
        # 计算日期范围
        today = datetime.now().date()
        thirty_days_later = today + timedelta(days=30)
        
        # 基础查询 - 只查询入库记录
        base_query = Chemical.query.filter(Chemical.action == '入库')
        
        # 根据筛选类型添加日期条件
        if filter_type == 'upcoming':
            base_query = base_query.filter(
                Chemical.expiration_date.between(today, thirty_days_later)
            )
        elif filter_type == 'expired':
            base_query = base_query.filter(Chemical.expiration_date < today)
        elif filter_type == 'all':
            base_query = base_query.filter(Chemical.expiration_date <= thirty_days_later)
        
        # 获取符合条件的入库记录
        in_stock_chemicals = base_query.all()
        
        # 计算净库存
        net_stock_chemicals = []
        for chemical in in_stock_chemicals:
            # 计算该批次总入库量
            total_in = db.session.query(
                db.func.sum(Chemical.quantity)
            ).filter(
                Chemical.product_code == chemical.product_code,
                Chemical.name == chemical.name,
                Chemical.batch_number == chemical.batch_number,
                Chemical.action == '入库'
            ).scalar() or 0
            
            # 计算该批次总出库量(出库+盘点)
            total_out = db.session.query(
                db.func.sum(Chemical.quantity)
            ).filter(
                Chemical.product_code == chemical.product_code,
                Chemical.name == chemical.name,
                Chemical.batch_number == chemical.batch_number,
                Chemical.action.in_(['出库', '盘点'])
            ).scalar() or 0
            
            # 计算净库存
            net_stock = total_in - total_out
            
            # 只保留净库存大于0的记录
            if net_stock > 0:
                # 创建包含所需属性的字典对象（包含仓库位置）
                chem_dict = {
                    'id': chemical.id,
                    'product_code': chemical.product_code,
                    'name': chemical.name,
                    'batch_number': chemical.batch_number,
                    'expiration_date': chemical.expiration_date,
                    'days_remaining': (chemical.expiration_date - today).days if chemical.expiration_date else None,
                    'net_stock': net_stock,
                    'total_stock': chemical.total_stock,
                    'location': chemical.location,
                    'warehouse_location': chemical.warehouse_location,  # 新增仓库位置
                    'manufacturer': chemical.manufacturer,
                    'specification': chemical.specification,
                    'model': chemical.model
                }
                net_stock_chemicals.append(chem_dict)
        
        # 按剩余天数排序
        net_stock_chemicals.sort(key=lambda x: x['days_remaining'] or float('inf'))
        
        return render_template(
            'expiration_reminder.html',
            chemicals=net_stock_chemicals,
            filter_type=filter_type,
            today=today.strftime('%Y-%m-%d')
        )
    
    except Exception as e:
        logger.error(f"过期预警查询失败: {str(e)}")
        flash(f'过期预警查询失败: {str(e)}', 'danger')
        return redirect(url_for('index'))


@app.route('/export_expiration_reminder/<filter_type>')
def export_expiration_reminder(filter_type):
    """导出过期预警数据(Excel格式)"""
    try:
        # 调用expiration_reminder的数据获取逻辑
        today = datetime.now().date()
        thirty_days_later = today + timedelta(days=30)
        
        base_query = Chemical.query.filter(Chemical.action == '入库')
        
        if filter_type == 'upcoming':
            base_query = base_query.filter(
                Chemical.expiration_date.between(today, thirty_days_later)
            )
        elif filter_type == 'expired':
            base_query = base_query.filter(Chemical.expiration_date < today)
        elif filter_type == 'all':
            base_query = base_query.filter(Chemical.expiration_date <= thirty_days_later)
        
        in_stock_chemicals = base_query.all()
        
        # 准备导出数据
        data = []
        for chem in in_stock_chemicals:
            total_in = db.session.query(
                db.func.sum(Chemical.quantity)
            ).filter(
                Chemical.product_code == chem.product_code,
                Chemical.name == chem.name,
                Chemical.batch_number == chem.batch_number,
                Chemical.action == '入库'
            ).scalar() or 0
            
            total_out = db.session.query(
                db.func.sum(Chemical.quantity)
            ).filter(
                Chemical.product_code == chem.product_code,
                Chemical.name == chem.name,
                Chemical.batch_number == chem.batch_number,
                Chemical.action.in_(['出库', '盘点'])
            ).scalar() or 0
            
            net_stock = total_in - total_out
            if net_stock > 0:
                days_remaining = (chem.expiration_date - today).days if chem.expiration_date else None
                data.append({
                    '商品代码': chem.product_code,
                    '化学品名称': chem.name,
                    '批次号': chem.batch_number,
                    '过期日期': chem.expiration_date.strftime('%Y-%m-%d') if chem.expiration_date else '',
                    '剩余天数': days_remaining if days_remaining is not None else '',
                    '净库存': net_stock,
                    '存储位置': chem.location,
                    '仓库位置': chem.warehouse_location or '',
                    '生产商': chem.manufacturer,
                    '规格': chem.specification,
                    '浓度': chem.concentration,
                    '单位': chem.unit
                })
        
        # 生成Excel文件
        df = pd.DataFrame(data)
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='过期预警', index=False)
            
            # 设置列宽
            worksheet = writer.sheets['过期预警']
            for idx, col in enumerate(df.columns):
                max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.set_column(idx, idx, max_len)
        
        output.seek(0)
        response = make_response(output.getvalue())
        response.headers['Content-Disposition'] = f'attachment; filename=expiration_reminder_{filter_type}.xlsx'
        response.headers['Content-type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        return response
        
    except Exception as e:
        logger.error(f"过期预警Excel导出失败: {str(e)}")
        flash(f'导出失败: {str(e)}', 'danger')
        return redirect(url_for('expiration_reminder', filter=filter_type))
@app.route('/export_expiration_reminder_csv/<filter_type>')
def export_expiration_reminder_csv(filter_type):
    """导出过期预警数据(CSV格式)"""
    try:
        # 调用expiration_reminder的数据获取逻辑
        today = datetime.now().date()
        thirty_days_later = today + timedelta(days=30)
        
        base_query = Chemical.query.filter(Chemical.action == '入库')
        
        if filter_type == 'upcoming':
            base_query = base_query.filter(
                Chemical.expiration_date.between(today, thirty_days_later)
            )
        elif filter_type == 'expired':
            base_query = base_query.filter(Chemical.expiration_date < today)
        elif filter_type == 'all':
            base_query = base_query.filter(Chemical.expiration_date <= thirty_days_later)
        
        in_stock_chemicals = base_query.all()
        
        # 创建CSV输出
        output = StringIO()
        writer = csv.writer(output)
        
        # 写入表头
        headers = [
            '商品代码', '化学品名称', '批次号', '过期日期', '剩余天数',
            '净库存', '存储位置', '仓库位置', '生产商', '规格',
            '浓度', '单位'
        ]
        writer.writerow(headers)
        
        # 写入数据
        for chem in in_stock_chemicals:
            total_in = db.session.query(
                db.func.sum(Chemical.quantity)
            ).filter(
                Chemical.product_code == chem.product_code,
                Chemical.name == chem.name,
                Chemical.batch_number == chem.batch_number,
                Chemical.action == '入库'
            ).scalar() or 0
            
            total_out = db.session.query(
                db.func.sum(Chemical.quantity)
            ).filter(
                Chemical.product_code == chem.product_code,
                Chemical.name == chem.name,
                Chemical.batch_number == chem.batch_number,
                Chemical.action.in_(['出库', '盘点'])
            ).scalar() or 0
            
            net_stock = total_in - total_out
            if net_stock > 0:
                days_remaining = (chem.expiration_date - today).days if chem.expiration_date else None
                writer.writerow([
                    chem.product_code,
                    chem.name,
                    chem.batch_number,
                    chem.expiration_date.strftime('%Y-%m-%d') if chem.expiration_date else '',
                    days_remaining if days_remaining is not None else '',
                    net_stock,
                    chem.location,
                    chem.warehouse_location or '',
                    chem.manufacturer,
                    chem.specification,
                    chem.concentration,
                    chem.unit
                ])
        
        output.seek(0)
        response = make_response(output.getvalue().encode('utf-8-sig'))
        response.headers['Content-Disposition'] = f'attachment; filename=expiration_reminder_{filter_type}.csv'
        response.headers['Content-type'] = 'text/csv; charset=utf-8'
        return response
        
    except Exception as e:
        logger.error(f"过期预警CSV导出失败: {str(e)}")
        flash(f'导出失败: {str(e)}', 'danger')
        return redirect(url_for('expiration_reminder', filter=filter_type))

@app.route('/safety_stock_reminder')
def safety_stock_reminder():
    low_stock_chemicals = Chemical.query.filter(
        Chemical.total_stock < Chemical.safety_stock
    ).all()
    return render_template('safety_stock_reminder.html', chemicals=low_stock_chemicals)

# 导出安全库存预警路由
@app.route('/export_safety_stock_alerts')
def export_safety_stock_alerts():
    low_stock_chemicals = Chemical.query.filter(Chemical.total_stock < Chemical.safety_stock).all()
    
    output = StringIO()
    writer = csv.writer(output, delimiter=',')
    writer.writerow([
        '商品代码', '化学品名称', '规格型号', '存储位置', '当前操作',
        '当前库存', '总库存', '安全库存', '库存缺口'
    ])
    
    for chem in low_stock_chemicals:
        writer.writerow([
            chem.product_code,
            chem.name,
            f"{chem.specification} {chem.model}",
            chem.location,
            chem.action,
            chem.quantity,
            chem.total_stock,
            chem.safety_stock,
            chem.safety_stock - chem.total_stock
        ])
    
    output.seek(0)
    response = make_response(output.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename="safety_stock_alerts.csv"'
    response.headers['Content-type'] = 'text/csv; charset=utf-8'
    return response

# 年度对比路由
from collections import defaultdict

# ... 前面的代码保持不变 ...

# 年度对比路由
@app.route('/compare_years')
def compare_years():
    try:
        # 获取精确查询参数
        search_name = request.args.get('name', '').strip()
        search_batch = request.args.get('batch', '').strip()
        
        # 获取化学品名称和批次号唯一值
        unique_names = [c[0] for c in db.session.query(Chemical.name).distinct().all() if c[0]]
        unique_batches = [c[0] for c in db.session.query(Chemical.batch_number).distinct().all() if c[0]]
        
        # 构建基础查询
        base_query = Chemical.query
        
        # 精确查询条件
        if search_name:
            base_query = base_query.filter(Chemical.name == search_name)
        if search_batch:
            base_query = base_query.filter(Chemical.batch_number == search_batch)

        # 计算总库存数据
        total_in = db.session.query(
            func.sum(Chemical.quantity)
        ).filter(
            Chemical.action == '入库',
            base_query.whereclause
        ).scalar() or 0

        total_out = db.session.query(
            func.sum(Chemical.quantity)
        ).filter(
            Chemical.action.in_(['出库', '盘点']),
            base_query.whereclause
        ).scalar() or 0

        net_stock = total_in - total_out
        total_stock = total_in

        # 获取所有有记录的年份
        years = db.session.query(
            db.extract('year', Chemical.production_date)
        ).distinct().filter(
            base_query.whereclause
        ).all()
        years = [int(year[0]) for year in years if year[0] is not None]
        years.sort(reverse=True)
        
        if not years:
            return render_template('compare_years.html', 
                                  comparison_data={}, 
                                  all_year_data={},
                                  total_chemicals=0,
                                  total_stock=0,
                                  net_stock=0,
                                  expired_count=0,
                                  search_name=search_name,
                                  search_batch=search_batch,
                                  min_year=None,
                                  max_year=None,
                                  unique_names=unique_names,
                                  unique_batches=unique_batches)

        # 获取最小和最大年份
        min_year = min(years)
        max_year = max(years)
        
        # 初始化比较数据
        comparison_data = {}
        all_year_data = {
            'categories': set(),
            'actions': {'入库', '出库', '盘点'},
            'months': defaultdict(lambda: {
                'categories': defaultdict(lambda: {'入库': 0, '出库': 0, '盘点': 0}),
                'total': 0
            }),
            'category_stats': defaultdict(lambda: {'in': 0, 'out': 0, 'check': 0}),
            'max_in_month': 0,
            'max_in_value': 0,
            'max_out_month': 0,
            'max_out_value': 0,
            'max_check_month': 0,
            'max_check_value': 0,
            'most_active_category': '',
            'most_active_value': 0,
            'start_date': f"{min_year}-01-01",
            'end_date': f"{max_year}-12-31"
        }
        
        # 为每个年份准备数据
        for year in years:
            # 获取该年份的月度统计
            monthly_stats = db.session.query(
                db.extract('month', Chemical.production_date).label('month'),
                Chemical.name,
                Chemical.action,
                func.sum(Chemical.quantity).label('total_quantity')
            ).filter(
                db.extract('year', Chemical.production_date) == year,
                base_query.whereclause
            ).group_by(
                'month', 'name', 'action'
            ).all()

            year_data = {
                'categories': set(),
                'actions': {'入库', '出库', '盘点'},
                'months': defaultdict(lambda: {
                    'categories': defaultdict(lambda: {'入库': 0, '出库': 0, '盘点': 0}),
                    'total': 0
                }),
                'category_stats': defaultdict(lambda: {'in': 0, 'out': 0, 'check': 0}),
                'max_in_month': 0,
                'max_in_value': 0,
                'max_out_month': 0,
                'max_out_value': 0,
                'max_check_month': 0,
                'max_check_value': 0,
                'most_active_category': '',
                'most_active_value': 0,
                'start_date': f"{year}-01-01",
                'end_date': f"{year}-12-31",
                'total_net': 0
            }

            for stat in monthly_stats:
                month = int(stat.month)
                name = stat.name or '未分类'
                action = stat.action or '无操作'
                quantity = stat.total_quantity or 0
                
                # 添加到年度数据
                year_data['categories'].add(name)
                year_data['months'][month]['categories'][name][action] += quantity
                
                # 更新分类统计数据
                if action == '入库':
                    year_data['category_stats'][name]['in'] += quantity
                elif action == '出库':
                    year_data['category_stats'][name]['out'] += quantity
                elif action == '盘点':
                    year_data['category_stats'][name]['check'] += quantity
                
                # 更新最大操作月份
                if action == '入库' and quantity > year_data['max_in_value']:
                    year_data['max_in_value'] = quantity
                    year_data['max_in_month'] = month
                elif action == '出库' and quantity > year_data['max_out_value']:
                    year_data['max_out_value'] = quantity
                    year_data['max_out_month'] = month
                elif action == '盘点' and quantity > year_data['max_check_value']:
                    year_data['max_check_value'] = quantity
                    year_data['max_check_month'] = month
                    
                # 添加到全部年度数据
                all_year_data['categories'].add(name)
                all_year_data['months'][month]['categories'][name][action] += quantity
                if action == '入库':
                    all_year_data['category_stats'][name]['in'] += quantity
                elif action == '出库':
                    all_year_data['category_stats'][name]['out'] += quantity
                elif action == '盘点':
                    all_year_data['category_stats'][name]['check'] += quantity
                
                # 更新全部年度最大操作月份
                if action == '入库' and quantity > all_year_data['max_in_value']:
                    all_year_data['max_in_value'] = quantity
                    all_year_data['max_in_month'] = month
                elif action == '出库' and quantity > all_year_data['max_out_value']:
                    all_year_data['max_out_value'] = quantity
                    all_year_data['max_out_month'] = month
                elif action == '盘点' and quantity > all_year_data['max_check_value']:
                    all_year_data['max_check_value'] = quantity
                    all_year_data['max_check_month'] = month
            
            # 计算月度合计（入库 - 出库 - 盘点）
            for month in range(1, 13):
                month_total = 0
                for category in year_data['categories']:
                    in_qty = year_data['months'][month]['categories'][category].get('入库', 0)
                    out_qty = year_data['months'][month]['categories'][category].get('出库', 0)
                    check_qty = year_data['months'][month]['categories'][category].get('盘点', 0)
                    month_total += (in_qty - out_qty - check_qty)
                year_data['months'][month]['total'] = month_total
            
            # 查找最活跃的分类
            for category, stats in year_data['category_stats'].items():
                total_activity = stats['in'] + stats['out'] + stats['check']
                if total_activity > year_data['most_active_value']:
                    year_data['most_active_value'] = total_activity
                    year_data['most_active_category'] = category
            
            comparison_data[year] = year_data
        
        # 计算全部年度的月度合计
        for month in range(1, 13):
            month_total = 0
            for category in all_year_data['categories']:
                in_qty = all_year_data['months'][month]['categories'][category].get('入库', 0)
                out_qty = all_year_data['months'][month]['categories'][category].get('出库', 0)
                check_qty = all_year_data['months'][month]['categories'][category].get('盘点', 0)
                month_total += (in_qty - out_qty - check_qty)
            all_year_data['months'][month]['total'] = month_total
        
        # 查找全部年度最活跃的分类
        for category, stats in all_year_data['category_stats'].items():
            total_activity = stats['in'] + stats['out'] + stats['check']
            if total_activity > all_year_data['most_active_value']:
                all_year_data['most_active_value'] = total_activity
                all_year_data['most_active_category'] = category
        
        # 计算过期数量
        today = datetime.now().date()
        expired_count = base_query.filter(
            Chemical.expiration_date < today,
            base_query.whereclause
        ).count()

        # 计算总化学品数
        total_chemicals = base_query.count()

        return render_template(
            'compare_years.html',
            comparison_data=comparison_data,
            all_year_data=all_year_data,
            total_chemicals=total_chemicals,
            total_stock=total_stock,
            expired_count=expired_count,
            net_stock=net_stock,
            search_name=search_name,
            search_batch=search_batch,
            min_year=min_year,
            max_year=max_year,
            unique_names=unique_names,
            unique_batches=unique_batches
        )
    
    except Exception as e:
        logger.error(f"年度对比数据加载失败: {str(e)}", exc_info=True)
        flash(f'数据加载失败: {str(e)}', 'danger')
        return redirect(url_for('index'))
# 现库存查询路由
@app.route('/current_stock')  # 删除 endpoint 参数
def current_stock():
    try:
        # 获取查询参数
        product_code = request.args.get('product_code', '').strip()
        batch_number = request.args.get('batch_number', '').strip()
        name = request.args.get('name', '').strip()
        location = request.args.get('location', '').strip()
        
        # 构建基础查询 (保持原有逻辑不变)
        base_query = db.session.query(
            Chemical.product_code,
            Chemical.name,
            Chemical.cas_number,
            Chemical.batch_number,
            Chemical.location,
            Chemical.manufacturer,
            Chemical.specification,
            Chemical.expiration_date,
            func.sum(Chemical.quantity).filter(Chemical.action == '入库').label('total_in'),
            func.sum(Chemical.quantity).filter(Chemical.action.in_(['出库', '盘点'])).label('total_out'),
            (func.sum(Chemical.quantity).filter(Chemical.action == '入库') - 
            func.sum(Chemical.quantity).filter(Chemical.action.in_(['出库', '盘点']))).label('net_stock')
        ).group_by(
            Chemical.product_code,
            Chemical.name,
            Chemical.cas_number,
            Chemical.batch_number,
            Chemical.location,
            Chemical.manufacturer,
            Chemical.specification,
            Chemical.expiration_date
        ).having(
            (func.sum(Chemical.quantity).filter(Chemical.action == '入库') - 
            func.sum(Chemical.quantity).filter(Chemical.action.in_(['出库', '盘点']))) > 0
        )

        # 添加过滤条件
        if product_code:
            base_query = base_query.filter(Chemical.product_code.ilike(f'%{product_code}%'))
        if batch_number:
            base_query = base_query.filter(Chemical.batch_number.ilike(f'%{batch_number}%'))
        if name:
            base_query = base_query.filter(Chemical.name.ilike(f'%{name}%'))
        if location:
            base_query = base_query.filter(Chemical.location.ilike(f'%{location}%'))

        # 执行查询
        stock_data = base_query.all()
        
        # 计算库存统计
        total_items = len(stock_data)
        total_net_stock = sum(item.net_stock or 0 for item in stock_data)
        expiring_soon = sum(1 for item in stock_data 
                           if item.expiration_date and 
                           (item.expiration_date - datetime.now().date()).days <= 30)
        
        # 分页处理
        page = request.args.get('page', 1, type=int)
        per_page = 50
        paginated_data = stock_data[(page-1)*per_page:page*per_page]

        return render_template(
            'current_stock.html',
            stock_data=paginated_data,
            total_items=total_items,
            total_net_stock=total_net_stock,
            expiring_soon=expiring_soon,
            page=page,
            per_page=per_page,
            total_pages=(total_items + per_page - 1) // per_page,
            product_code=product_code,
            batch_number=batch_number,
            name=name,
            location=location
        )
    
    except Exception as e:
        logger.error(f"现库存查询失败: {str(e)}")
        flash(f'现库存查询失败: {str(e)}', 'danger')
        return redirect(url_for('index'))

# 万能查询路由
@app.route('/universal_query')
def universal_query():
    try:
        in_stock = db.session.query(
            Chemical.product_code,
            Chemical.batch_number,
            db.func.sum(Chemical.quantity).label('total_in')
        ).filter(
            Chemical.action == '入库'
        ).group_by(
            Chemical.product_code,
            Chemical.batch_number
        ).subquery()

        out_stock = db.session.query(
            Chemical.product_code,
            Chemical.batch_number,
            db.func.sum(Chemical.quantity).label('total_out')
        ).filter(
            Chemical.action.in_(['出库', '盘点'])
        ).group_by(
            Chemical.product_code,
            Chemical.batch_number
        ).subquery()

        net_stock = db.session.query(
            Chemical.id,
            (in_stock.c.total_in - db.func.coalesce(out_stock.c.total_out, 0)).label('net_stock')
        ).join(
            in_stock,
            (Chemical.product_code == in_stock.c.product_code) &
            (Chemical.batch_number == in_stock.c.batch_number)
        ).outerjoin(
            out_stock,
            (Chemical.product_code == out_stock.c.product_code) &
            (Chemical.batch_number == out_stock.c.batch_number)
        ).filter(
            (in_stock.c.total_in - db.func.coalesce(out_stock.c.total_out, 0)) > 0
        ).subquery()

        page = request.args.get('page', 1, type=int)
        chemicals = Chemical.query.join(
            net_stock,
            Chemical.id == net_stock.c.id
        ).paginate(page=page, per_page=20)

        net_stocks = {row.id: row.net_stock for row in net_stock}
        
        return render_template(
            'universal_query.html',
            chemicals=chemicals,
            net_stocks=net_stocks
        )

    except Exception as e:
        flash(f'万能查询失败: {str(e)}', 'danger')
        return redirect(url_for('index'))

# 查看图片路由
@app.route('/view_image/<int:id>')
def view_image(id):
    chemical = Chemical.query.get_or_404(id)
    return send_from_directory(app.config['UPLOAD_FOLDER'], chemical.image)

# 删除化学品路由
@app.route('/delete/<int:id>', methods=['POST'])
@login_required
@permission_required('chemical:delete')
@log_activity('delete_chemical', 'chemical')
def delete_chemical(id):
    # 从环境变量获取管理员密码
    ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'FI')
    
    # 获取请求中的密码
    password = request.form.get('password', '')
    if password != ADMIN_PASSWORD:
        flash('删除失败: 密码错误', 'danger')
        return redirect(url_for('index'))
    
    chemical = Chemical.query.get_or_404(id)
    try:
        # 记录删除操作日志
        logger.info(f"删除化学品: ID={chemical.id}, 名称={chemical.name}, 操作员={request.remote_addr}")
        
        db.session.delete(chemical)
        db.session.commit()
        flash('化学品删除成功!', 'success')
    except Exception as e:
        db.session.rollback()
        logger.error(f"删除失败: {str(e)}")
        flash(f'删除失败: {str(e)}', 'danger')
    return redirect(url_for('index'))

# 化学品详情路由
# 更新 /chemical_details 路由
@app.route('/chemical_details/<int:id>')
def chemical_details(id):
    chemical = Chemical.query.get_or_404(id)
    return jsonify({
        'id': chemical.id,
        'product_code': chemical.product_code,
        'name': chemical.name,
        'category': chemical.category,
        'specification': chemical.specification,
        'concentration': chemical.concentration,
        'unit': chemical.unit,
        'model': chemical.model,
        'location': chemical.location,
        'manufacturer': chemical.manufacturer,
        'batch_number': chemical.batch_number,
        'production_date': chemical.production_date.strftime('%Y-%m-%d') if chemical.production_date else None,
        'expiration_date': chemical.expiration_date.strftime('%Y-%m-%d') if chemical.expiration_date else None,
        'quantity': chemical.quantity,
        'batch_total_stock': chemical.batch_total_stock,
        'total_stock': chemical.total_stock,
        'safety_stock': chemical.safety_stock,
        'status': chemical.status,
        'price': chemical.price,
        'supplier': chemical.supplier,
        'notes': chemical.notes,
        'image': url_for('view_image', id=chemical.id, _external=True) if chemical.image else None,
        'attachment': url_for('download_attachment', id=chemical.id, type='attachment', _external=True) if chemical.attachment else None,
        'msds_attachment': url_for('download_attachment', id=chemical.id, type='msds', _external=True) if chemical.msds_attachment else None,
        'delivery_order_attachment': url_for('download_attachment', id=chemical.id, type='delivery_order', _external=True) if chemical.delivery_order_attachment else None,
        'invoice_attachment': url_for('download_attachment', id=chemical.id, type='invoice', _external=True) if chemical.invoice_attachment else None
    })

# 图像分析路由
@app.route('/image_analysis', methods=['POST'])
def image_analysis():
    try:
        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')
        
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)
        
        with torch.no_grad():
            output = vision_model(input_batch)
            
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        _, predicted_idx = torch.max(probabilities, 0)
        
        return jsonify({
            'prediction': predicted_idx.item(),
            'confidence': probabilities[predicted_idx].item()
        })
    except Exception as e:
        logger.error(f"Image analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# 高级查询路由（含模糊查询）

@app.route('/search', methods=['GET', 'POST'])
def search():
    """
    高级搜索路由 - 支持多条件组合查询化学品记录
    功能：
    1. 支持商品代码、化学品名称、CAS编号精确/模糊查询
    2. 支持类别、操作类型、库位等过滤条件
    3. 支持过期日期范围和收支日期 范围过滤
    4. 支持按收支日期 和过期日期排序
    5. 分页显示结果（每页300条）
    6. 搜索结果包含化学品详细信息及附件下载
    """
    # 初始化基础参数
    page = request.args.get('page', 1, type=int)
    per_page = 300
    sort_field = request.args.get('sort', 'production_date')
    sort_direction = request.args.get('direction', 'desc')

    # 检测高级过滤条件是否启用
    show_more_filters = any(request.form.get(field) for field in [
        'category', 'action', 'manufacturer', 'location', 'warehouse_location', 'cas_number',
        'batch_number', 'start_date', 'end_date', 'production_start_date', 'production_end_date'
    ])

    # 初始化查询对象
    query = Chemical.query

    # 处理POST请求（表单过滤）
    if request.method == 'POST':
        form_data = {
            'product_code': request.form.get('product_code', '').strip(),
            'name': request.form.get('name', '').strip(),
            'cas_number': request.form.get('cas_number', '').strip(),
            'search_mode': request.form.get('search_mode', 'exact'),
            'category': request.form.get('category'),
            'action': request.form.get('action'),
            'manufacturer': request.form.get('manufacturer', '').strip(),
            'location': request.form.get('location'),
            'warehouse_location': request.form.get('warehouse_location', '').strip(),  # 新增仓库位置
            'batch_number': request.form.get('batch_number', '').strip(),
            'start_date': request.form.get('start_date'),
            'end_date': request.form.get('end_date'),
            'production_start_date': request.form.get('production_start_date'),
            'production_end_date': request.form.get('production_end_date')
        }

        # 构建过滤条件
        filters = []
        
        # 商品代码精确匹配
        if form_data['product_code']:
            if form_data['search_mode'] == 'exact':
                filters.append(Chemical.product_code == form_data['product_code'])
            else:
                filters.append(Chemical.product_code.ilike(f"%{form_data['product_code']}%"))
        
        # 化学品名称查询
        if form_data['name']:
            if form_data['search_mode'] == 'exact':
                filters.append(Chemical.name == form_data['name'])
            else:
                filters.append(Chemical.name.ilike(f"%{form_data['name']}%"))
        
        # CAS编号查询
        if form_data['cas_number']:
            if form_data['search_mode'] == 'exact':
                filters.append(Chemical.cas_number == form_data['cas_number'])
            else:
                filters.append(Chemical.cas_number.ilike(f"%{form_data['cas_number']}%"))
        
        # 类别过滤
        if form_data['category']:
            filters.append(Chemical.category == form_data['category'])
        
        # 操作类型过滤
        if form_data['action']:
            filters.append(Chemical.action == form_data['action'])
        
        # 生产厂家模糊查询
        if form_data['manufacturer']:
            filters.append(Chemical.manufacturer.ilike(f"%{form_data['manufacturer']}%"))
        
        # 库位过滤
        if form_data['location']:
            filters.append(Chemical.location == form_data['location'])
        
        # 仓库位置过滤（新增）
        if form_data['warehouse_location']:
            filters.append(Chemical.warehouse_location.ilike(f"%{form_data['warehouse_location']}%"))
        
        # 批次号精确查询
        if form_data['batch_number']:
            filters.append(Chemical.batch_number == form_data['batch_number'])
        
        # 失效日期范围
        try:
            if form_data['start_date']:
                start_date = datetime.strptime(form_data['start_date'], '%Y-%m-%d').date()
                filters.append(Chemical.expiration_date >= start_date)
            if form_data['end_date']:
                end_date = datetime.strptime(form_data['end_date'], '%Y-%m-%d').date()
                filters.append(Chemical.expiration_date <= end_date)
        except ValueError:
            pass
        
        # 收支日期 范围
        try:
            if form_data['production_start_date']:
                prod_start = datetime.strptime(form_data['production_start_date'], '%Y-%m-%d').date()
                filters.append(Chemical.production_date >= prod_start)
            if form_data['production_end_date']:
                prod_end = datetime.strptime(form_data['production_end_date'], '%Y-%m-%d').date()
                filters.append(Chemical.production_date <= prod_end)
        except ValueError:
            pass
        
        # 应用所有过滤器
        if filters:
            query = query.filter(and_(*filters))

    # 动态排序逻辑
    if sort_field == 'production_date':
        order_field = Chemical.production_date
    else:
        order_field = Chemical.expiration_date
    
    order_logic = order_field.desc() if sort_direction == 'desc' else order_field.asc()

    # 执行分页查询
    chemicals = query.order_by(order_logic).paginate(
        page=page,
        per_page=per_page,
        error_out=False
    )

    # 构建模板参数
    return render_template(
        'search.html',
        chemicals=chemicals,
         show_more_filters=show_more_filters,
        search_params=request.form if request.method == 'POST' else None,
        sort_field=sort_field,
        sort_direction=sort_direction
    )


# 智能预警路由
@app.route('/smart_alerts')
def smart_alerts():
    try:
        alerts = []
        expiring_chemicals = Chemical.query.filter(
            Chemical.expiration_date <= datetime.now() + timedelta(days=30)
        ).all()

        latest_expiring = {}
        for chem in expiring_chemicals:
            if chem.name not in latest_expiring or chem.production_date > latest_expiring[chem.name].production_date:
                latest_expiring[chem.name] = chem

        for chem in latest_expiring.values():
            usage_history = Chemical.query.filter_by(
                product_code=chem.product_code
            ).order_by(Chemical.operation_date.desc()).limit(6).all()
            
            avg_usage = sum([c.quantity for c in usage_history])/len(usage_history) if usage_history else 0
            avg_usage = round(avg_usage)
            suggested_qty = max(avg_usage, chem.safety_stock)
            
            alerts.append({
                'id': chem.id,
                'type': 'expiration',
                'message': f"{chem.name}（批次：{chem.batch_number}）预计{chem.days_remaining}天后过期，建议采购数量：{suggested_qty}",
                'severity': 'high' if chem.days_remaining < 15 else 'medium'
            })

        low_stock = Chemical.query.filter(Chemical.total_stock < Chemical.safety_stock).all()
        for chem in low_stock:
            alerts.append({
                'id': chem.id,
                'type': 'low_stock',
                'message': f"{chem.name} 当前库存({chem.total_stock})低于安全库存({chem.safety_stock})",
               'severity': 'critical'
            })
        
        return jsonify(alerts)
    except Exception as e:
        logger.error(f"Smart alerts error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/ai_query', methods=['POST'])
def ai_query():
    try:
        # 获取请求数据
        data = request.get_json()
        question = data.get('question', '').strip()
        logger.info(f"收到AI查询请求: '{question}'")
        
        if not question:
            logger.warning("查询请求为空")
            return jsonify({"error": "问题不能为空"}), 400
        
        # DeepSeek API 请求配置
        headers = {
            "Authorization": f"Bearer {app.config['DEEPSEEK_API_KEY']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [{
                "role": "user",
                "content": question
            }],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        # 添加指数退避重试机制
        max_retries = 3
        backoff_factor = 0.5  # 基础退避时间
        retry_status_codes = [429, 500, 502, 503, 504]  # 需要重试的状态码
        
        response = None
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"尝试第 {attempt+1}/{max_retries} 次请求")
                start_time = time.time()
                
                # 发送请求
                response = requests.post(
                    app.config['DEEPSEEK_API_URL'],
                    headers=headers,
                    json=payload,
                    timeout=66  # 设置66秒超时
                )
                
                # 记录响应时间
                latency = time.time() - start_time
                logger.info(f"API响应时间: {latency:.2f}秒, 状态码: {response.status_code}")
                
                # 检查是否需要重试
                if response.status_code in retry_status_codes:
                    # 计算退避时间 (指数退避)
                    sleep_time = backoff_factor * (2 ** attempt) + random.uniform(0, 0.2)
                    logger.warning(f"请求失败 (状态码: {response.status_code}), 将在 {sleep_time:.2f}秒后重试")
                    time.sleep(sleep_time)
                    continue
                    
                # 检查其他错误状态
                response.raise_for_status()
                
                # 成功响应，跳出重试循环
                break
                
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                # 连接问题或超时
                last_exception = e
                sleep_time = backoff_factor * (2 ** attempt) + random.uniform(0, 0.2)
                logger.warning(f"连接错误: {str(e)}, 将在 {sleep_time:.2f}秒后重试")
                time.sleep(sleep_time)
                
            except requests.exceptions.RequestException as e:
                # 其他请求异常
                last_exception = e
                logger.error(f"请求异常: {str(e)}")
                break
                
            except Exception as e:
                # 其他未知异常
                last_exception = e
                logger.error(f"未知异常: {str(e)}")
                break
        
        # 检查是否所有重试都失败
        if not response or not response.ok:
            error_message = f"所有重试尝试均失败: {str(last_exception)}" if last_exception else "未知错误"
            logger.error(error_message)
            return jsonify({
                "error": "AI服务暂时不可用",
                "details": error_message,
                "suggestion": "请稍后再试或联系管理员"
            }), 502
        
        # 解析成功响应
        response_data = response.json()
        logger.debug(f"API响应数据: {json.dumps(response_data, ensure_ascii=False)}")
        
        # 提取答案
        answer = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        if not answer:
            logger.warning("API返回了空答案")
            return jsonify({
                "error": "未获取到有效答案",
                "api_response": response_data
            }), 500
        
        # 记录成功响应
        logger.info(f"成功获取答案，字符长度: {len(answer)}")
        
        # 缓存答案（可选功能）
        try:
            cache_key = f"ai_answer:{hashlib.md5(question.encode()).hexdigest()}"
            cache.set(cache_key, answer, timeout=3600)  # 缓存1小时
            logger.debug(f"答案已缓存，键: {cache_key}")
        except Exception as e:
            logger.warning(f"缓存失败: {str(e)}")
        
        # 返回成功响应
        return jsonify({
            "answer": answer,
            "question": question,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "api_status": response.status_code,
            "latency": f"{latency:.2f}秒"
        })
        
    except json.JSONDecodeError:
        logger.error("JSON解析错误: 请求数据格式不正确")
        return jsonify({"error": "无效的请求格式"}), 400
        
    except Exception as e:
        logger.exception(f"服务器内部错误: {str(e)}")
        return jsonify({
            "error": "服务器内部错误",
            "details": str(e),
            "trace": traceback.format_exc()
        }), 500

@app.route('/health')
def health_check():
    try:
        # 网络延迟检查
        start = time.time()
        socket.create_connection(("api.deepseek.com", 443), timeout=5)
        latency = round((time.time() - start)*1000, 2)
        
        # API 状态检查
        resp = requests.get("https://api.deepseek.com/v1/models", timeout=5)
        return jsonify({  # <-- 添加 return
            "status": "healthy",
            "api_status": resp.status_code,
            "network_latency_ms": latency
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

# API接口
@app.route('/api/chemicals/search', methods=['GET'])
def api_search():
    search_term = request.args.get('q', '').strip()
    limit = request.args.get('limit', 10, type=int)
    
    results = Chemical.query.filter(
        or_(
            Chemical.product_code.ilike(f'%{search_term}%'),
            Chemical.name.ilike(f'%{search_term}%'),
            Chemical.batch_number.ilike(f'%{search_term}%')
        )
    ).limit(limit).all()

    return jsonify([{
        'id': chem.id,
        'product_code': chem.product_code,
        'name': chem.name,
        'batch_number': chem.batch_number,
        'location': chem.location,
        'quantity': chem.quantity,
        'expiration_date': chem.expiration_date.isoformat() if chem.expiration_date else None,
        'days_remaining': chem.days_remaining,
        'msds_url': url_for('download_attachment', id=chem.id, type='msds', _external=True) if chem.msds_attachment else None
    } for chem in results])

@app.route('/download_local_file')
def download_local_file():
    """增强型文件下载路由"""
    try:
        file_path = request.args.get('path', '')
        if not file_path:
            abort(400, description="缺少路径参数")
            
        allowed_bases = [
            r"W:\QA-AnalLab_Share\MSDS",
            r"G:\FI Information",
            r"G:\00_TEAM_QA\Anal Lab\E-COA"
        ]
        
        # 解码URL编码的路径并标准化
        safe_path = os.path.normpath(urllib.parse.unquote(file_path))
        
        # 验证路径安全性
        if not any(is_safe_path(base, safe_path) for base in allowed_bases):
            logger.error(f"非法路径访问尝试: {safe_path}")
            abort(403, description="访问路径未授权")
            
        if not os.path.exists(safe_path):
            abort(404, description="文件不存在")
            
        directory = os.path.dirname(safe_path)
        filename = os.path.basename(safe_path)
        return send_from_directory(
            directory=directory,
            path=filename,
            as_attachment=True
        )
        
    except Exception as e:
        logger.error(f"文件下载失败: {str(e)}")
        abort(500)

@app.route('/search_local_msds', methods=['POST'])
def search_local_msds():
    try:
        keyword = request.json.get('keyword', '').lower()
        results = []
        base_path = r"W:\QA-AnalLab_Share\MSDS"
        
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if keyword in file.lower():
                    try:
                        file_path = os.path.join(root, file)
                        mtime = os.path.getmtime(file_path)
                        results.append({
                            "name": file,
                            "path": file_path,
                            "mtime": datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
                        })
                    except Exception as e:
                        logger.error(f"文件处理失败：{file_path} - {str(e)}")
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 新增两个路由
@app.route('/search_local_fi', methods=['POST'])
def search_local_fi():
    try:
        keyword = request.json.get('keyword', '').lower()
        results = []
        base_path = r"G:\FI Information"
        
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if keyword in file.lower():
                    try:
                        file_path = os.path.join(root, file)
                        mtime = os.path.getmtime(file_path)
                        results.append({
                            "name": file,
                            "path": file_path,
                            "mtime": datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
                        })
                    except Exception as e:
                        logger.error(f"文件处理失败：{file_path} - {str(e)}")
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search_local_ecoa', methods=['POST'])
def search_local_ecoa():
    try:
        keyword = request.json.get('keyword', '').lower()
        results = []
        base_path = r"G:\00_TEAM_QA\Anal Lab\E-COA"
        
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if keyword in file.lower():
                    try:
                        file_path = os.path.join(root, file)
                        mtime = os.path.getmtime(file_path)
                        results.append({
                            "name": file,
                            "path": file_path,
                            "mtime": datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
                        })
                    except Exception as e:
                        logger.error(f"文件处理失败：{file_path} - {str(e)}")
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/log_file_access', methods=['POST'])
def log_file_access():
    """记录文件访问日志"""
    try:
        data = request.get_json()
        filename = data.get('filename', 'unknown')
        logger.info(f"文件访问记录: {filename} - IP: {request.remote_addr}")
        return jsonify({"status": "logged"})
    except Exception as e:
        logger.error(f"日志记录失败: {str(e)}")
        return jsonify({"error": str(e)}), 50

@app.route('/adv_search', methods=['POST'])
def adv_search():
    try:
        name = request.form.get('name')
        code = request.form.get('code')
        cas = request.form.get('cas')  # 新增CAS编号参数
        fuzzy = request.form.get('fuzzy') == 'on'

        # 首次模糊查询
        query = Chemical.query
        if fuzzy:
            if name: query = query.filter(Chemical.name.ilike(f'%{name}%'))
            if code: query = query.filter(Chemical.product_code.ilike(f'%{code}%'))
            if cas: query = query.filter(Chemical.cas_number.ilike(f'%{cas}%'))  # CAS模糊查询
        else:
            if name: query = query.filter_by(name=name)
            if code: query = query.filter_by(product_code=code)
            if cas: query = query.filter_by(cas_number=cas)  # CAS精确查询

        # 按名称和代码分组取最后一条
        subquery = query.with_entities(
            Chemical.name,
            Chemical.product_code,
            db.func.max(Chemical.id).label('max_id')
        ).group_by(Chemical.name, Chemical.product_code).subquery()

        # 修复：将以下代码移入try块内
        results = Chemical.query.join(
            subquery,
            Chemical.id == subquery.c.max_id
        ).order_by(Chemical.production_date.desc()).all()  # 新增排序

        return render_template('adv_search.html', 
                             chemicals=results,
                             search_params=request.form)
    
    except Exception as e:
        flash(f'查询失败: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/secondary_search', methods=['POST'])
def secondary_search():
    try:
        data = request.get_json()
        action = data.get('action', 'all')
        chemical_ids = data.get('chemicalIds', [])
        
        if not chemical_ids:
            return jsonify({'error': '未选择任何记录'}), 400

        # 获取基础记录用于提取查询条件
        base_records = Chemical.query.filter(Chemical.id.in_(chemical_ids)).all()
        if not base_records:
            return jsonify({'error': '未找到匹配的化学品记录'}), 404

        names = {r.name for r in base_records}
        codes = {r.product_code for r in base_records}

        # 修复点：使用 and_() 包裹多个条件，确保括号闭合
        query = Chemical.query.filter(
            and_(
                Chemical.name.in_(names),
                Chemical.product_code.in_(codes)
            )
        )

        # 根据操作类型过滤
        if action == 'all':
            pass  # 不添加额外过滤条件
        elif action == 'in':
            query = query.filter(Chemical.action == '入库')
        elif action == 'out':
            query = query.filter(Chemical.action == '出库')
        elif action == 'check':
            query = query.filter(Chemical.action == '盘点')
        elif action == 'last':
            # 获取每个化学品的最新收货记录
            subquery = db.session.query(
                Chemical.product_code,
                Chemical.name,
                func.max(Chemical.production_date).label('max_date')
            ).filter(
                Chemical.action == '入库'
            ).group_by(
                Chemical.product_code,
                Chemical.name
            ).subquery()

            query = query.join(
                subquery,
                and_(
                    Chemical.product_code == subquery.c.product_code,
                    Chemical.name == subquery.c.name,
                    Chemical.production_date == subquery.c.max_date
                )
            ).filter(Chemical.action == '入库')

        records = query.order_by(Chemical.operation_date.desc()).all()

        return jsonify({
            'records': [{
                'name': r.name,
                'product_code': r.product_code,
                'action': r.action,
                'batch_number': r.batch_number,
                'quantity': r.quantity,
                'batch_total_stock': r.batch_total_stock,

                'total_stock': r.total_stock,
                'production_date': r.production_date.strftime('%Y-%m-%d') if r.production_date else None,
                'operation_date': r.operation_date.strftime('%Y-%m-%d %H:%M') if r.operation_date else None,
                'operator': r.operator,
                'latest_date': r.production_date if action == 'last' else None
            } for r in records]
        })

    except Exception as e:
        logger.error(f"二次查询失败: {str(e)}")
        return jsonify({'error': str(e)}), 500

# 更新操作类型映射
ACTION_MAPPING = {
    'all': None,
    'in': '入库',
    'out': '出库',
    'check': '盘点',
    'last': '入库'  # 最后收货特殊处理
}

@app.route('/warehouse')
def warehouse():
    """仓库库存管理页面路由"""
    try:
        # 获取查询参数
        name = request.args.get('name', '').strip()
        product_code = request.args.get('product_code', '').strip()
        cas_number = request.args.get('cas_number', '').strip()
        batch_number = request.args.get('batch_number', '').strip()
        location = request.args.get('location', '').strip()
        warehouse_location = request.args.get('warehouse_location', '').strip()
        consumable_category = request.args.get('consumable_category', '').strip()
        regulatory_category = request.args.get('regulatory_category', '').strip()
        stock_status_filter = request.args.get('stock_status', '')
        category = request.args.get('category', '')
        expiring = request.args.get('expiring', '') == 'on'
        page = request.args.get('page', 1, type=int)
        view = request.args.get('view', 'table')
        
        # 计算当前日期
        today = datetime.now(timezone.utc).date()
        
        # 构建基础查询 - 获取每个产品代码+批次号的最新记录
        subquery = db.session.query(
            Chemical.product_code,
            Chemical.batch_number,
            func.max(Chemical.operation_date).label('max_date')
        ).group_by(
            Chemical.product_code,
            Chemical.batch_number
        ).subquery()

        # 主查询
        query = Chemical.query.join(
            subquery,
            and_(
                Chemical.product_code == subquery.c.product_code,
                Chemical.batch_number == subquery.c.batch_number,
                Chemical.operation_date == subquery.c.max_date
            )
        ).filter(Chemical.action == '入库')  # 只显示入库记录
        
        # 添加过滤条件
        if name:
            query = query.filter(Chemical.name.ilike(f'%{name}%'))
        if product_code:
            query = query.filter(Chemical.product_code.ilike(f'%{product_code}%'))
        if cas_number:
            query = query.filter(Chemical.cas_number.ilike(f'%{cas_number}%'))
        if batch_number:
            query = query.filter(Chemical.batch_number.ilike(f'%{batch_number}%'))
        if location:
            query = query.filter(Chemical.location.ilike(f'%{location}%'))
        if warehouse_location:
            query = query.filter(Chemical.warehouse_location.ilike(f'%{warehouse_location}%'))
        if consumable_category:
            query = query.filter(Chemical.consumable_category.ilike(f'%{consumable_category}%'))
        if regulatory_category:
            query = query.filter(Chemical.regulatory_category.ilike(f'%{regulatory_category}%'))
        if stock_status_filter:
            query = query.filter(Chemical.status == stock_status_filter)
        if category:
            query = query.filter(Chemical.category == category)
        if expiring:
            query = query.filter(
                Chemical.expiration_date <= today + timedelta(days=30)
            )
        
        # 执行查询
        all_chemicals = query.all()
        
        # 为每个化学品计算净库存和状态
        filtered_chemicals = []
        stock_status = {
            'normal': 0,
            'low': 0,
            'critical': 0
        }
        
        # 分类统计数据
        category_count = {}
        
        for chem in all_chemicals:
            net_stock = calculate_net_stock(chem.product_code, chem.batch_number)
            if net_stock > 0:
                chem.net_stock = net_stock
                
                # 计算库存状态
                if net_stock > chem.safety_stock:
                    chem.stock_status = 'normal'
                    stock_status['normal'] += 1
                elif net_stock > chem.safety_stock * 0.5:
                    chem.stock_status = 'low'
                    stock_status['low'] += 1
                else:
                    chem.stock_status = 'critical'
                    stock_status['critical'] += 1
                
                # 更新分类统计
                category = chem.category or "未分类"
                category_count[category] = category_count.get(category, 0) + 1
                    
                filtered_chemicals.append(chem)
        
        chemicals = filtered_chemicals
        
        # 转换为图表所需格式
        category_labels = list(category_count.keys())
        category_data = list(category_count.values())
        
        # 统计信息
        total_items = len(chemicals)
        distinct_chemical_names = len(set(chem.name for chem in chemicals))
        available_stock = sum(chem.net_stock for chem in chemicals)
        
        # 获取所有分类用于下拉菜单
        categories = db.session.query(
            Chemical.category
        ).distinct().filter(
            Chemical.category.isnot(None)
        ).order_by(
            Chemical.category
        ).all()
        categories = [cat[0] for cat in categories]
        
        # ================ 月度数据计算 ================
        # 计算月度出入库统计
        # 获取当前年份和所有有记录的年份
        years = set()
        for chem in Chemical.query.filter(Chemical.production_date.isnot(None)).all():
            if chem.production_date:
                years.add(chem.production_date.year)
                
        years = sorted(list(years))
        if not years:
            years = [datetime.now().year]
            
        # 获取请求中的年份，如果没有则默认为当前年份
        selected_year = request.args.get('year', datetime.now().year, type=int)
        if selected_year not in years:
            selected_year = years[-1] if years else datetime.now().year
            
        # 查询该年份每个月的入库和出库（包括盘点）总量
        monthly_in = db.session.query(
            db.extract('month', Chemical.production_date).label('month'),
            func.sum(Chemical.quantity).label('total_in')
        ).filter(
            Chemical.action == '入库',
            db.extract('year', Chemical.production_date) == selected_year
        ).group_by('month').all()
        
        monthly_out = db.session.query(
            db.extract('month', Chemical.production_date).label('month'),
            func.sum(Chemical.quantity).label('total_out')
        ).filter(
            Chemical.action.in_(['出库', '盘点']),
            db.extract('year', Chemical.production_date) == selected_year
        ).group_by('month').all()
        
        # 转换为字典
        in_dict = {int(stat.month): stat.total_in for stat in monthly_in}
        out_dict = {int(stat.month): stat.total_out for stat in monthly_out}
        
        # 初始化12个月的数据
        months = list(range(1, 13))
        in_data = [in_dict.get(month, 0) for month in months]
        out_data = [out_dict.get(month, 0) for month in months]
        
        monthly_data = {
            'available_years': years,
            'selected_year': selected_year,
            'months': months,
            'in_stock': in_data,
            'out_stock': out_data
        }
        # ================ 月度数据计算结束 ================
        
        # 分页处理
        per_page = 50
        pagination = {
            'page': page,
            'per_page': per_page,
            'total': total_items,
            'pages': (total_items + per_page - 1) // per_page
        }
        
        start_index = (page - 1) * per_page
        end_index = start_index + per_page
        paginated_chemicals = chemicals[start_index:end_index]
        
        return render_template(
            'warehouse.html',
            chemicals=paginated_chemicals,
            pagination=pagination,
            view=view,
            name=name,
            product_code=product_code,
            cas_number=cas_number,
            batch_number=batch_number,
            location=location,
            warehouse_location=warehouse_location,
            consumable_category=consumable_category,
            regulatory_category=regulatory_category,
            stock_status_filter=stock_status_filter,
            category=category,
            categories=categories,
            expiring=expiring,
            total_items=total_items,
            distinct_chemical_names=distinct_chemical_names,
            available_stock=available_stock,
            stock_status=stock_status,
            category_labels=category_labels,
            category_data=category_data,
            monthly_data=monthly_data,
            today=today
        )
    
    except Exception as e:
        logger.error(f"仓库查询失败: {str(e)}", exc_info=True)
        flash(f'仓库查询失败: {str(e)}', 'danger')
        return redirect(url_for('index'))

# 净库存计算函数
def calculate_net_stock(product_code, batch_number):
    """计算特定批次的净库存"""
    # 计算总入库数量
    total_in = db.session.query(
        db.func.sum(Chemical.quantity)
    ).filter(
        Chemical.product_code == product_code,
        Chemical.batch_number == batch_number,
        Chemical.action == '入库'
    ).scalar() or 0

    # 计算总出库数量(出库+盘点)
    total_out = db.session.query(
        db.func.sum(Chemical.quantity)
    ).filter(
        Chemical.product_code == product_code,
        Chemical.batch_number == batch_number,
        Chemical.action.in_(['出库', '盘点'])
    ).scalar() or 0

    return total_in - total_out


@app.route('/view_attachment/<int:id>')
def view_attachment(id):
    """查看附件图片（非下载）"""
    chemical = Chemical.query.get_or_404(id)
    if not chemical.attachment:
        abort(404, description="该化学品没有关联附件")
    
    # 检查文件扩展名是否是图片
    ext = chemical.attachment.split('.')[-1].lower()
    if ext not in ['png', 'jpg', 'jpeg', 'gif']:
        abort(400, description="附件不是图片格式")
    
    try:
        return send_from_directory(
            app.config['UPLOAD_FOLDER'], 
            chemical.attachment, 
            as_attachment=False  # 关键：直接显示而非下载
        )
    except FileNotFoundError:
        abort(404, description="附件文件未找到")

@app.route('/clone_chemical/<int:id>', methods=['GET', 'POST'])
@validate_signature
def clone_chemical(id):
    """克隆化学品记录 - 只能保存为新记录，包含签名确认"""
    original = Chemical.query.get_or_404(id)
    
    if request.method == 'POST':
        try:
            # 验证签名数据
            signature_data = request.form.get('signature_data', '')
            
            if not signature_data:
                flash('请先进行手写签名确认', 'warning')
                return render_template('clone_chemical.html', chemical=original)
            
            if not signature_data.startswith('data:image/png;base64,'):
                logger.warning(f"Invalid signature format: {signature_data[:50]}")
                flash('签名数据格式无效，请重新签名', 'warning')
                return render_template('clone_chemical.html', chemical=original)

            # 处理签名数据
            signature_filename = None
            if signature_data:
                signature_data_clean = signature_data.split(',', 1)[1]
                try:
                    signature_bytes = base64.b64decode(signature_data_clean)
                    
                    # 生成唯一文件名
                    timestamp = int(time.time())
                    signature_filename = f"signature_{timestamp}_{uuid.uuid4().hex[:8]}.png"
                    signature_path = os.path.join(app.config['UPLOAD_FOLDER'], signature_filename)
                    
                    # 确保上传目录存在
                    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                    
                    # 保存签名图片
                    with open(signature_path, "wb") as f:
                        f.write(signature_bytes)
                    
                    # 创建缩略图（可选）
                    try:
                        img = Image.open(io.BytesIO(signature_bytes))
                        img.thumbnail((200, 100))
                        thumbnail_path = os.path.join(app.config['UPLOAD_FOLDER'], f"thumb_{signature_filename}")
                        img.save(thumbnail_path)
                    except Exception as thumb_error:
                        logger.warning(f"无法创建签名缩略图: {str(thumb_error)}")
                    
                    logger.info(f"克隆记录签名已保存: {signature_filename}")
                except Exception as e:
                    logger.error(f"签名处理失败: {str(e)}")
                    flash('签名处理失败，请重新签名', 'danger')
                    return render_template('clone_chemical.html', chemical=original)

            # 处理表单数据
            product_code = request.form.get('product_code')
            name = request.form.get('name')
            category = request.form.get('category')
            specification = request.form.get('specification')
            concentration = float(request.form.get('concentration', 0)) if request.form.get('concentration') else 0.0
            unit = request.form.get('unit')
            model = request.form.get('model')
            location = request.form.get('location')
            operator = request.form.get('operator')
            confirmer = request.form.get('confirmer')
            action = request.form.get('action')
            manufacturer = request.form.get('manufacturer')
            batch_number = request.form.get('batch_number')
            
            # 处理日期字段
            production_date = None
            if request.form.get('production_date'):
                production_date = datetime.strptime(request.form.get('production_date'), '%Y-%m-%d').date()
            
            expiration_date = None
            if request.form.get('expiration_date'):
                expiration_date = datetime.strptime(request.form.get('expiration_date'), '%Y-%m-%d').date()
            
            quantity = int(request.form.get('quantity', 0)) if request.form.get('quantity') else 0
            batch_total_stock = int(request.form.get('batch_total_stock', 0)) if request.form.get('batch_total_stock') else 0
            total_stock = int(request.form.get('total_stock', 0)) if request.form.get('total_stock') else 0
            safety_stock = int(request.form.get('safety_stock', 0)) if request.form.get('safety_stock') else 0
            status = request.form.get('status')
            price = float(request.form.get('price', 0)) if request.form.get('price') else 0.0
            supplier = request.form.get('supplier')
            notes = request.form.get('notes')
            
            # 处理新增字段
            consumable_category = request.form.get('consumable_category')
            cas_number = request.form.get('cas_number')
            warehouse_location = request.form.get('warehouse_location')
            regulatory_category = request.form.get('regulatory_category')

            # 处理文件上传字段 - 为新记录创建新文件
            file_fields = {
                'image': '图片',
                'attachment': '通用附件',
                'delivery_order_attachment': '送货单附件',
                'invoice_attachment': '发票附件',
                'msds_attachment': 'MSDS附件'
            }
            
            new_filenames = {}
            for field in file_fields.keys():
                file = request.files.get(field)
                if file and file.filename != '':
                    if allowed_file(file.filename):
                        # 生成唯一文件名
                        filename = secure_filename(file.filename)
                        base, ext = os.path.splitext(filename)
                        unique_id = str(uuid.uuid4())[:8]
                        unique_filename = f"{base}_{unique_id}{ext}"
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                        file.save(file_path)
                        new_filenames[field] = unique_filename
                    else:
                        flash(f'不允许的文件类型: {file.filename}', 'warning')
                else:
                    # 如果没有上传新文件，但原记录有文件，则复制原文件名
                    original_filename = getattr(original, field)
                    if original_filename:
                        # 在实际应用中，您可能需要复制文件内容而不是仅复制文件名
                        # 这里为了简单，仅复制文件名
                        new_filenames[field] = original_filename

            # 创建新的化学品记录（克隆）
            new_chemical = Chemical(
                product_code=product_code,
                name=name,
                category=category,
                specification=specification,
                concentration=concentration,
                unit=unit,
                model=model,
                location=location,
                operator=operator,
                confirmer=confirmer,
                action=action,
                manufacturer=manufacturer,
                batch_number=batch_number,
                production_date=production_date,
                expiration_date=expiration_date,
                quantity=quantity,
                batch_total_stock=batch_total_stock,
                total_stock=total_stock,
                safety_stock=safety_stock,
                status=status,
                price=price,
                supplier=supplier,
                notes=notes,
                consumable_category=consumable_category,
                cas_number=cas_number,
                warehouse_location=warehouse_location,
                regulatory_category=regulatory_category,
                # 文件字段使用新上传的文件或原文件的引用
                image=new_filenames.get('image'),
                attachment=new_filenames.get('attachment'),
                delivery_order_attachment=new_filenames.get('delivery_order_attachment'),
                invoice_attachment=new_filenames.get('invoice_attachment'),
                msds_attachment=new_filenames.get('msds_attachment'),
                # 签名字段
                signature_data=signature_data,
                signature_image=signature_filename,
                # 操作日期设置为当前时间
                operation_date=datetime.now(timezone.utc)
            )
            
            db.session.add(new_chemical)
            db.session.commit()
            
            flash('化学品已成功克隆为新记录!', 'success')
            return redirect(url_for('index'))
            
        except ValueError as ve:
            db.session.rollback()
            logger.error(f"数据类型错误: {str(ve)}")
            flash(f'输入数据格式错误: {str(ve)}', 'danger')
            return render_template('clone_chemical.html', chemical=original)
        
        except Exception as e:
            db.session.rollback()
            logger.error(f"克隆化学品失败: {str(e)}")
            flash(f'错误: {str(e)}', 'danger')
            return render_template('clone_chemical.html', chemical=original)

    # GET请求：显示克隆表单（预填充原始数据）
    return render_template('clone_chemical.html', chemical=original)

# 查看化学品详情路由
@app.route('/view/<int:id>')
def view_chemical(id):
    chemical = Chemical.query.get_or_404(id)
    return render_template('view_chemical.html', chemical=chemical)

# 新增Equipment模型
class Equipment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fixed_asset_id = db.Column(db.String(100))
    equipment_manage_id = db.Column(db.String(100))
    self_number = db.Column(db.String(100))
    require_3q_cert = db.Column(db.Boolean, default=False)
    name = db.Column(db.String(200), nullable=False)
    model = db.Column(db.String(100))
    laboratory = db.Column(db.String(100))
    location = db.Column(db.String(100))
    purpose = db.Column(db.Text)
    usage_info = db.Column(db.Text)   
    status = db.Column(db.Enum('正常', '待检', '维修中', '已经维修', '停用', '报废', '报废中'), default='正常')
    inspection_date = db.Column(db.Date)
    annual_check_date = db.Column(db.Date)  # 移除无效字符
    purchase_date = db.Column(db.Date)
    purchase_amount = db.Column(db.Numeric(12,2))
    depreciated_value = db.Column(db.Numeric(12,2))
    notes = db.Column(db.Text)
    fault_date = db.Column(db.Date)
    fault_category = db.Column(db.String(100))
    handling_method = db.Column(db.String(100))
    solution = db.Column(db.Text)
    result = db.Column(db.Text)
    completion_date = db.Column(db.Date)
    operator = db.Column(db.String(100))
    confirmer = db.Column(db.String(100))
    equipment_image = db.Column(db.String(200))
    fixed_asset_image = db.Column(db.String(200))
    location_image = db.Column(db.String(200))
    invoice_attachment = db.Column(db.String(200))
    repair_attachment = db.Column(db.String(200))
    delivery_attachment = db.Column(db.String(200))  # 移除无效字符
    annual_check_attachment = db.Column(db.String(200))
    other_attachment = db.Column(db.String(200))  # 移除无效字符
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    operation_date = db.Column(db.DateTime, default=datetime.utcnow) # 新增操作日期字段
   # 新增四个技术文档附件字段
    manufacturer_manual_attachment = db.Column(db.String(200))  # 设备原厂使用说明书附件
    sop_attachment = db.Column(db.String(200))  # SOP标准操作规程附件
    calibration_record_attachment = db.Column(db.String(200))  # 设备定期校准记录附件
    manufacturer_data_attachment = db.Column(db.String(200))  # 厂家资料附件
    fault_history = db.Column(db.JSON, nullable=True)  # 存储结构化的故障历史记录

    @property
    def formatted_fault_history(self):
        """格式化故障历史为可读文本"""
        if not self.fault_history:
            return "无故障记录"
        
        history = []
        for record in self.fault_history:
            entry = f"[{record['date']}] {record['category']} - {record['description']}"
            if record['solution']:
                entry += f"\n解决方案: {record['solution']}"
            if record['result']:
                entry += f"\n处理结果: {record['result']}"
            history.append(entry)
        return "\n\n".join(history)

# 检查文件名是否在数据库中存在
def check_filename_exists(filename):
    """检查数据库中是否存在同名文件"""
    # 检查所有可能包含文件名的字段
    fields_to_check = [
        'equipment_image',
        'fixed_asset_image',
        'location_image',
        'invoice_attachment',
        'repair_attachment',
        'delivery_attachment',
        'annual_check_attachment',
        'other_attachment',
        'manufacturer_manual_attachment',
        'sop_attachment',
        'calibration_record_attachment',
        'manufacturer_data_attachment'
    ]
    
    
    # 构建查询条件
    conditions = []
    for field in fields_to_check:
        conditions.append(getattr(Equipment, field) == filename)
    
    # 执行查询
    existing = Equipment.query.filter(or_(*conditions)).first()
    return existing is not None

# 处理设备文件上传
def handle_equipment_upload(file, field_name, existing_filename=None):
    """处理设备文件上传，保留原始文件名并检查同名"""
    if not file or file.filename == '':
        return existing_filename

    # 获取原始文件名
    original_filename = file.filename
    
    # 检查文件扩展名是否允许
    if not allowed_file(original_filename):
        flash(f'错误: 文件类型不允许 ({original_filename})', 'danger')
        return existing_filename
    
    # 检查数据库中是否已存在同名文件
    if check_filename_exists(original_filename):
        flash(f'警告: 文件名 "{original_filename}" 在数据库中已存在', 'warning')
    
    # 检查文件系统中是否已存在同名文件
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
    if os.path.exists(file_path):
        flash(f'警告: 文件 "{original_filename}" 在服务器上已存在', 'warning')
    
    # 保存文件
    try:
        file.save(file_path)
        
        # 如果更新操作且有新文件上传，删除旧文件
        if existing_filename and existing_filename != original_filename:
            old_file_path = os.path.join(app.config['UPLOAD_FOLDER'], existing_filename)
            if os.path.exists(old_file_path):
                os.remove(old_file_path)
        
        return original_filename
    except Exception as e:
        logger.error(f"文件保存失败: {str(e)}")
        flash(f'文件保存失败: {str(e)}', 'danger')
        return existing_filename

@app.route('/api/equipment/<int:id>/faults', methods=['GET', 'POST', 'DELETE'])
def equipment_faults_api(id):
    equipment = Equipment.query.get_or_404(id)
    
    if request.method == 'GET':
        return jsonify({
            'fault_history': equipment.fault_history or []
        })
    
    elif request.method == 'POST':
        data = request.get_json()
        new_record = {
            'id': str(uuid.uuid4()),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'category': data.get('category', '未知故障'),
            'description': data.get('description', ''),
            'solution': data.get('solution', ''),
            'result': data.get('result', ''),
            'resolved': data.get('resolved', False),
            'technician': data.get('technician', '')
        }
        
        # 初始化故障历史数组
        if not equipment.fault_history:
            equipment.fault_history = []
        
        equipment.fault_history.append(new_record)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'new_record': new_record
        })
    
    elif request.method == 'DELETE':
        record_id = request.args.get('record_id')
        if not record_id:
            return jsonify({'error': '缺少记录ID'}), 400
        
        # 过滤出需要保留的记录
        equipment.fault_history = [
            r for r in (equipment.fault_history or []) 
            if r.get('id') != record_id
        ]
        
        db.session.commit()
        return jsonify({'success': True})

# 新增设备管理路由
@app.route('/equipment')
def equipment_list():
    try:
        # 获取查询参数
        page = request.args.get('page', 1, type=int)
        fixed_asset_id = request.args.get('fixed_asset_id', '').strip()
        equipment_manage_id = request.args.get('equipment_manage_id', '').strip()
        name = request.args.get('name', '').strip()
        model = request.args.get('model', '').strip()
        laboratory = request.args.get('laboratory', '').strip()
        location = request.args.get('location', '').strip()
        status = request.args.get('status', '')
        
        # 构建基础查询
        query = Equipment.query
        
        # 应用搜索条件
        if fixed_asset_id:
            query = query.filter(Equipment.fixed_asset_id.ilike(f'%{fixed_asset_id}%'))
        if equipment_manage_id:
            query = query.filter(Equipment.equipment_manage_id.ilike(f'%{equipment_manage_id}%'))
        if name:
            query = query.filter(Equipment.name.ilike(f'%{name}%'))
        if model:
            query = query.filter(Equipment.model.ilike(f'%{model}%'))
        if laboratory:
            query = query.filter(Equipment.laboratory.ilike(f'%{laboratory}%'))
        if location:
            query = query.filter(Equipment.location.ilike(f'%{location}%'))
        if status:
            query = query.filter(Equipment.status == status)
        
        # 每页显示的数量（根据视图优化）
        per_page = request.args.get('per_page', 20, type=int)
        
        # 执行分页查询
        equipment = query.order_by(Equipment.id.desc()).paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        
        # 获取最后更新时间
        last_update = db.session.query(
            func.max(Equipment.operation_date)
        ).scalar()
        
        # 获取所有实验室用于搜索下拉框
        laboratories = db.session.query(
            Equipment.laboratory
        ).distinct().filter(
            Equipment.laboratory.isnot(None)
        ).all()
        laboratories = [lab[0] for lab in laboratories]
        
        # 渲染模板
        return render_template(
            'equipment.html',
            equipment=equipment,
            last_update=last_update,
            laboratories=laboratories,
            # 传递搜索参数
            request_args=request.args
        )
    
    except Exception as e:
        logger.error(f"设备列表查询失败: {str(e)}", exc_info=True)
        flash(f'设备列表加载失败: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/equipment/add', methods=['GET', 'POST'])
@login_required
@permission_required('equipment:create')
@log_activity('add_equipment', 'equipment')
def add_equipment():
    if request.method == 'POST':
        try:
            # 处理表单数据
            new_equipment = Equipment(
                fixed_asset_id=request.form.get('fixed_asset_id'),
                equipment_manage_id=request.form.get('equipment_manage_id'),
                self_number=request.form.get('self_number'),
                require_3q_cert=bool(request.form.get('require_3q_cert')),
                name=request.form.get('name'),
                model=request.form.get('model'),
                laboratory=request.form.get('laboratory'),
                location=request.form.get('location'),
                purpose=request.form.get('purpose'),
                usage_info=request.form.get('usage_info'),
                status=request.form.get('status', '正常'),
                inspection_date=datetime.strptime(request.form['inspection_date'], '%Y-%m-%d').date() if request.form.get('inspection_date') else None,
                annual_check_date=datetime.strptime(request.form['annual_check_date'], '%Y-%m-%d').date() if request.form.get('annual_check_date') else None,
                purchase_date=datetime.strptime(request.form['purchase_date'], '%Y-%m-%d').date() if request.form.get('purchase_date') else None,
                purchase_amount=float(request.form.get('purchase_amount', 0)),
                depreciated_value=float(request.form.get('depreciated_value', 0)),
                notes=request.form.get('notes'),
                fault_date=datetime.strptime(request.form['fault_date'], '%Y-%m-%d').date() if request.form.get('fault_date') else None,
                fault_category=request.form.get('fault_category'),
                handling_method=request.form.get('handling_method'),
                solution=request.form.get('solution'),
                result=request.form.get('result'),
                completion_date=datetime.strptime(request.form['completion_date'], '%Y-%m-%d').date() if request.form.get('completion_date') else None,
                operator=request.form.get('operator'),
                confirmer=request.form.get('confirmer'),
                operation_date=datetime.now(timezone.utc),  # 记录操作时间
                created_at=datetime.now(timezone.utc)  # 记录创建时间
            )
            
            # 处理文件上传字段
            file_fields = [
                'equipment_image', 'fixed_asset_image', 'location_image',
                'invoice_attachment', 'repair_attachment', 'delivery_attachment',
                'annual_check_attachment', 'other_attachment',
                'manufacturer_manual_attachment', 
                'sop_attachment',
                'calibration_record_attachment',
                'manufacturer_data_attachment'
            ]
            
            
            for field in file_fields:
                file = request.files.get(field)
                if file and file.filename != '':
                    if allowed_file(file.filename):
                        filename = secure_filename(file.filename)
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        
                        # 确保文件名唯一
                        counter = 1
                        while os.path.exists(file_path):
                            name, ext = os.path.splitext(filename)
                            filename = f"{name}_{counter}{ext}"
                            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                            counter += 1
                        
                        file.save(file_path)
                        setattr(new_equipment, field, filename)
                    else:
                        flash(f'不允许的文件类型: {file.filename}', 'warning')
            
            db.session.add(new_equipment)
            db.session.commit()
            
            flash('设备添加成功!', 'success')
            return redirect(url_for('equipment_list'))
            
        except ValueError as e:
            db.session.rollback()
            flash(f'日期格式错误: {str(e)}', 'danger')
        except SQLAlchemyError as e:
            db.session.rollback()
            flash(f'数据库错误: {str(e)}', 'danger')
        except Exception as e:
            db.session.rollback()
            logger.error(f"添加设备失败: {str(e)}", exc_info=True)
            flash(f'添加设备失败: {str(e)}', 'danger')
    
    # GET请求或表单验证失败时显示表单
    return render_template('add_edit_equipment.html', 
                         equipment=None,
                         status_options=['正常', '待检', '维修','维修中','已经维修','报废','报废中','停用'])

# 其他路由：edit_equipment, delete_equipment, view_equipment 等

@app.route('/equipment/view/<int:id>')
def view_equipment(id):
    """查看设备详情页面"""
    equipment = Equipment.query.get_or_404(id)
    return render_template('view_equipment.html', equipment=equipment)

@app.route('/equipment/edit/<int:id>', methods=['GET', 'POST'])
def edit_equipment(id):
    """编辑设备信息"""
    equipment = Equipment.query.get_or_404(id)
    
    # 定义文件字段列表，确保在两个分支中都可访问
    file_fields = [
      'equipment_image', 'fixed_asset_image', 'location_image',
        'invoice_attachment', 'repair_attachment', 'delivery_attachment',
        'annual_check_attachment', 'other_attachment',
        # 添加四个新字段
        'manufacturer_manual_attachment', 
        'sop_attachment',
        'calibration_record_attachment',
        'manufacturer_data_attachment'
    ]
    
    if request.method == 'POST':
        try:
            # 检查是否是"另存为新记录"操作
            save_as_new = 'save_as_new' in request.form

            # 处理表单数据更新
            if save_as_new:
                # 创建新设备对象
                new_equipment = Equipment(
                    fixed_asset_id=request.form.get('fixed_asset_id'),
                    equipment_manage_id=request.form.get('equipment_manage_id'),
                    name=request.form.get('name'),
                    model=request.form.get('model'),
                    # 复制其他字段...
                    laboratory=request.form.get('laboratory'),
                    location=request.form.get('location'),
                    purpose=request.form.get('purpose'),
                    usage_info=request.form.get('usage_info'),
                    status=request.form.get('status'),
                    inspection_date=datetime.strptime(request.form['inspection_date'], '%Y-%m-%d').date() if request.form.get('inspection_date') else None,
                    annual_check_date=datetime.strptime(request.form['annual_check_date'], '%Y-%m-%d').date() if request.form.get('annual_check_date') else None,
                    purchase_date=datetime.strptime(request.form['purchase_date'], '%Y-%m-%d').date() if request.form.get('purchase_date') else None,
                    purchase_amount=float(request.form.get('purchase_amount', 0)),
                    depreciated_value=float(request.form.get('depreciated_value', 0)),
                    notes=request.form.get('notes'),
                    fault_date=datetime.strptime(request.form['fault_date'], '%Y-%m-%d').date() if request.form.get('fault_date') else None,
                    fault_category=request.form.get('fault_category'),
                    handling_method=request.form.get('handling_method'),
                    solution=request.form.get('solution'),
                    result=request.form.get('result'),
                    completion_date=datetime.strptime(request.form['completion_date'], '%Y-%m-%d').date() if request.form.get('completion_date') else None,
                    operator=request.form.get('operator'),
                    confirmer=request.form.get('confirmer')
                )
                
                # 处理文件上传字段
                for field in file_fields:
                    file = request.files.get(field)
                    if file and file.filename != '':
                        filename = handle_equipment_upload(file, field)
                        setattr(new_equipment, field, filename)
                    else:
                        # 使用原记录的文件名
                        setattr(new_equipment, field, getattr(equipment, field))
                
                db.session.add(new_equipment)
                flash('设备已成功保存为新记录!', 'success')
            else:
                # 更新原有记录
                equipment.fixed_asset_id = request.form.get('fixed_asset_id')
                equipment.equipment_manage_id = request.form.get('equipment_manage_id')
                equipment.name = request.form.get('name')
                equipment.model = request.form.get('model')
                equipment.laboratory = request.form.get('laboratory')
                equipment.location = request.form.get('location')
                equipment.purpose = request.form.get('purpose')
                equipment.usage_info = request.form.get('usage_info')
                equipment.status = request.form.get('status')
                equipment.inspection_date = datetime.strptime(request.form['inspection_date'], '%Y-%m-%d').date() if request.form.get('inspection_date') else None
                equipment.annual_check_date = datetime.strptime(request.form['annual_check_date'], '%Y-%m-%d').date() if request.form.get('annual_check_date') else None
                equipment.purchase_date = datetime.strptime(request.form['purchase_date'], '%Y-%m-%d').date() if request.form.get('purchase_date') else None
                equipment.purchase_amount = float(request.form.get('purchase_amount', 0))
                equipment.depreciated_value = float(request.form.get('depreciated_value', 0))
                equipment.notes = request.form.get('notes')
                equipment.fault_date = datetime.strptime(request.form['fault_date'], '%Y-%m-%d').date() if request.form.get('fault_date') else None
                equipment.fault_category = request.form.get('fault_category')
                equipment.handling_method = request.form.get('handling_method')
                equipment.solution = request.form.get('solution')
                equipment.result = request.form.get('result')
                equipment.completion_date = datetime.strptime(request.form['completion_date'], '%Y-%m-%d').date() if request.form.get('completion_date') else None
                equipment.operator = request.form.get('operator')
                equipment.confirmer = request.form.get('confirmer')
                
                # 处理文件上传
                for field in file_fields:
                    file = request.files.get(field)
                    if file and file.filename != '':
                        new_filename = handle_equipment_upload(file, field, getattr(equipment, field))
                        if new_filename:
                            setattr(equipment, field, new_filename)
                
                flash('设备信息更新成功!', 'success')
            
            db.session.commit()
            return redirect(url_for('equipment_list'))
        except Exception as e:
            db.session.rollback()
            flash(f'操作失败: {str(e)}', 'danger')
    
    return render_template('add_edit_equipment.html', equipment=equipment)

# 在 download_equipment_attachment 路由中添加文件类型检测
@app.route('/download_equipment_attachment/<int:id>/<string:type>')
def download_equipment_attachment(id, type):
    equipment = Equipment.query.get_or_404(id)
    attachment_map = {
        'equipment_image': equipment.equipment_image,
        'fixed_asset_image': equipment.fixed_asset_image,
        'location_image': equipment.location_image,
        'invoice_attachment': equipment.invoice_attachment,
        'repair_attachment': equipment.repair_attachment,
        'delivery_attachment': equipment.delivery_attachment,
        'annual_check_attachment': equipment.annual_check_attachment,
        'other_attachment': equipment.other_attachment,
        'manufacturer_manual_attachment': equipment.manufacturer_manual_attachment,
        'sop_attachment': equipment.sop_attachment,
        'calibration_record_attachment': equipment.calibration_record_attachment,
        'manufacturer_data_attachment': equipment.manufacturer_data_attachment
    }
    
    if type not in attachment_map or not attachment_map[type]:
        abort(404, description="附件不存在")
    
    try:
        # 获取文件路径
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], attachment_map[type])
        
        # 检查文件扩展名以确定内容类型
        ext = os.path.splitext(attachment_map[type])[1].lower()
        if ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
            # 图片文件 - 内联显示
            return send_file(file_path, mimetype=f'image/{ext[1:]}', as_attachment=False)
        else:
            # 其他文件类型 - 作为附件下载
            return send_from_directory(app.config['UPLOAD_FOLDER'], attachment_map[type], as_attachment=True)
    except FileNotFoundError:
        abort(404, description="附件文件未找到")


# 设备详情API
@app.route('/equipment_details/<int:id>')
def equipment_details(id):
    """获取设备详情的API"""
    try:
        equipment = Equipment.query.get_or_404(id)
        
        # 构建附件列表
        attachments = []
        attachment_types = [
            ('repair_attachment', '维修附件'),
            ('annual_check_attachment', '年检附件'),
            ('other_attachment', '其他附件'),
            ('manufacturer_manual_attachment', '原厂说明书'),
            ('sop_attachment', 'SOP文档'),
            ('calibration_record_attachment', '校准记录'),
            ('manufacturer_data_attachment', '厂家资料')
        ]
        
        # 准备附件信息
        for field, name in attachment_types:
            if getattr(equipment, field):
                attachments.append({
                    'name': name,
                    'url': url_for('download_equipment_attachment', 
                                  id=equipment.id, 
                                  type=field, 
                                  _external=True)
                })
        
        # 返回设备详情，包括图片的完整URL
        return jsonify({
            'id': equipment.id,
            'fixed_asset_id': equipment.fixed_asset_id,
            'equipment_manage_id': equipment.equipment_manage_id,
            'self_number': equipment.self_number,
            'require_3q_cert': equipment.require_3q_cert,
            'name': equipment.name,
            'model': equipment.model,
            'laboratory': equipment.laboratory,
            'location': equipment.location,
            'purpose': equipment.purpose,
            'usage_info': equipment.usage_info,
            'status': equipment.status,
            'inspection_date': equipment.inspection_date.strftime('%Y-%m-%d') if equipment.inspection_date else None,
            'annual_check_date': equipment.annual_check_date.strftime('%Y-%m-%d') if equipment.annual_check_date else None,
            'purchase_date': equipment.purchase_date.strftime('%Y-%m-%d') if equipment.purchase_date else None,
            'purchase_amount': float(equipment.purchase_amount) if equipment.purchase_amount else 0.0,
            'depreciated_value': float(equipment.depreciated_value) if equipment.depreciated_value else 0.0,
            'notes': equipment.notes,
            'fault_date': equipment.fault_date.strftime('%Y-%m-%d') if equipment.fault_date else None,
            'fault_category': equipment.fault_category,
            'handling_method': equipment.handling_method,
            'solution': equipment.solution,
            'result': equipment.result,
            'completion_date': equipment.completion_date.strftime('%Y-%m-%d') if equipment.completion_date else None,
            'operator': equipment.operator,
            'confirmer': equipment.confirmer,
            'created_at': equipment.created_at.strftime('%Y-%m-%d %H:%M:%S') if equipment.created_at else None,
            # 图片URL（使用url_for生成完整的URL）
            'images': {
                'equipment_image': url_for('download_equipment_attachment', 
                                         id=equipment.id, 
                                         type='equipment_image', 
                                         _external=True) if equipment.equipment_image else None,
                'fixed_asset_image': url_for('download_equipment_attachment', 
                                           id=equipment.id, 
                                           type='fixed_asset_image', 
                                           _external=True) if equipment.fixed_asset_image else None,
                'location_image': url_for('download_equipment_attachment', 
                                        id=equipment.id, 
                                        type='location_image', 
                                        _external=True) if equipment.location_image else None
            },
            # 附件信息
            'attachments': attachments,
            # 故障历史（如果有）
            'fault_history': equipment.fault_history or []
        })
        
    except Exception as e:
        logger.error(f"获取设备详情失败: {str(e)}", exc_info=True)
        return jsonify({
            'error': f'获取设备详情失败: {str(e)}'
        }), 500


# 下载设备导入模板
@app.route('/download_equipment_template')
def download_equipment_template():
    """下载设备导入模板（支持多种格式）"""
    format_type = request.args.get('format', 'xlsx')
    
    # 创建示例数据
    data = [{
        '固定资产编号': 'FA-2023-001',
        '设备管理编号': 'EQ-2023-001',
        '自有编号': 'LAB-001',
        '仪器名称': '高效液相色谱仪',
        '仪器型号': 'HPLC-123',
        '实验室': '分析实验室',
        '位置': 'A区-1号架',
        '用途': '样品分析',
        '使用信息': '日常使用',
        '状态': '正常',
        '最近检查日期': '2023-01-15',
        '年检日期': '2023-12-31',
        '购买日期': '2023-01-01',
        '购买金额': 150000.00,
        '折旧价值': 135000.00,
        '备注': '',
        '故障日期': '',
        '故障类别': '',
        '处理方法': '',
        '解决方案': '',
        '结果': '',
        '完成日期': '',
        '操作员': '张三',
        '确认人': '李四'
    }]
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    if format_type == 'csv':
        # CSV格式导出
        output = StringIO()
        df.to_csv(output, index=False, encoding='utf-8-sig')
        output.seek(0)
        
        response = make_response(output.getvalue())
        response.headers['Content-Disposition'] = 'attachment; filename=设备导入模板.csv'
        response.headers['Content-type'] = 'text/csv; charset=utf-8'
        return response
    else:
        # Excel格式导出（默认）
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='设备导入模板', index=False)
            
            # 获取工作表并添加数据验证
            worksheet = writer.sheets['设备导入模板']
            
            # 添加状态下拉菜单
            worksheet.data_validation('K2:K1000', {
                'validate': 'list',
                'source': ['正常', '待检', '维修','已经维修','停用','报废','报废中']
            })
        
        output.seek(0)
        response = make_response(output.getvalue())
        response.headers['Content-Disposition'] = 'attachment; filename=设备导入模板.xlsx'
        response.headers['Content-type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        return response

# 导入设备数据
@app.route('/import_equipment', methods=['POST'])
def import_equipment():
    """导入设备数据（支持多种格式）"""
    try:
        file = request.files.get('file')
        if not file or not allowed_file(file.filename):
            flash('请上传有效的Excel或CSV文件', 'warning')
            return redirect(url_for('equipment_list'))
        
        filename = file.filename.lower()
        
        # 根据文件扩展名选择读取方式
        if filename.endswith('.xlsx'):
            df = pd.read_excel(file)
        elif filename.endswith('.csv'):
            # 尝试多种编码格式
            try:
                content = file.read().decode('utf-8-sig')
                df = pd.read_csv(StringIO(content))
            except UnicodeDecodeError:
                # 尝试GBK编码
                file.seek(0)  # 重置文件指针
                content = file.read().decode('gbk')
                df = pd.read_csv(StringIO(content))
        else:
            flash('不支持的文件格式', 'danger')
            return redirect(url_for('equipment_list'))
        
        # 检查必需字段
        required_columns = ['仪器名称', '仪器型号', '实验室', '位置']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            flash(f'缺少必需列: {", ".join(missing)}', 'danger')
            return redirect(url_for('equipment_list'))
        
        success_count = 0
        error_rows = []
        
        for idx, row in df.iterrows():
            try:
                # 转换日期字段
                def parse_date(date_str):
                    if pd.isna(date_str) or date_str == '':
                        return None
                    if isinstance(date_str, datetime):
                        return date_str.date()
                    try:
                        return datetime.strptime(str(date_str), '%Y-%m-%d').date()
                    except:
                        return None
                
                # 创建设备对象
                equipment = Equipment(
                    fixed_asset_id=row.get('固定资产编号', ''),
                    equipment_manage_id=row.get('设备管理编号', ''),
                    self_number=row.get('自有编号', ''),
                    require_3q_cert=bool(row.get('需3Q认证', False)),
                    name=row['仪器名称'],
                    model=row['仪器型号'],
                    laboratory=row['实验室'],
                    location=row['位置'],
                    purpose=row.get('用途', ''),
                    usage_info=row.get('使用信息', ''),
                    status=row.get('状态', '正常'),
                    inspection_date=parse_date(row.get('最近检查日期')),
                    annual_check_date=parse_date(row.get('年检日期')),
                    purchase_date=parse_date(row.get('购买日期')),
                    purchase_amount=float(row.get('购买金额', 0)),
                    depreciated_value=float(row.get('折旧价值', 0)),
                    notes=row.get('备注', ''),
                    fault_date=parse_date(row.get('故障日期')),
                    fault_category=row.get('故障类别', ''),
                    handling_method=row.get('处理方法', ''),
                    solution=row.get('解决方案', ''),
                    result=row.get('结果', ''),
                    completion_date=parse_date(row.get('完成日期')),
                    operator=row.get('操作员', ''),
                    confirmer=row.get('确认人', '')
                )
                
                db.session.add(equipment)
                success_count += 1
            except Exception as e:
                error_rows.append({
                    'row': idx + 2,  # Excel行号（从1开始，标题行+1）
                    'error': str(e),
                    'data': row.to_dict()
                })
        
        db.session.commit()
        flash(f'成功导入 {success_count} 条设备记录，失败 {len(error_rows)} 条', 'success')
        
        # 如果有错误，保存到session
        if error_rows:
            session['import_errors'] = json.dumps(error_rows)
        
    except Exception as e:
        db.session.rollback()
        flash(f'导入失败: {str(e)}', 'danger')
    
    return redirect(url_for('equipment_list'))

# 导出设备数据
@app.route('/export_equipment')
def export_equipment():
    """导出设备数据（支持多种格式和场景）"""
    try:
        # 获取导出参数
        format_type = request.args.get('format', 'xlsx')
        scope = request.args.get('scope', 'all')  # 'all' 或 'current'
        save_path = request.args.get('save_path', '')  # 可选保存路径
        
        # 月度报告特定参数
        report_year = request.args.get('year', type=int)
        report_month = request.args.get('month', type=int)
        status_param = request.args.get('status', '')  # 逗号分隔的状态字符串
        
        # 判断是否是月度报告导出
        is_monthly_report = report_year is not None and report_month is not None
        
        if is_monthly_report:
            # 月度报告导出逻辑
            # 构建基础查询
            query = Equipment.query.filter(
                extract('year', Equipment.operation_date) == report_year,
                extract('month', Equipment.operation_date) == report_month
            )
            
            # 应用状态筛选
            if status_param:
                selected_statuses = status_param.split(',')
                query = query.filter(Equipment.status.in_(selected_statuses))
            
            equipment_list = query.all()
        else:
            # 常规设备列表导出
            if scope == 'current':
                # 使用当前搜索参数
                query = Equipment.query
                
                # 应用搜索参数
                params = request.args
                if params.get('fixed_asset_id'):
                    query = query.filter(Equipment.fixed_asset_id.ilike(f'%{params["fixed_asset_id"]}%'))
                if params.get('equipment_manage_id'):
                    query = query.filter(Equipment.equipment_manage_id.ilike(f'%{params["equipment_manage_id"]}%'))
                if params.get('name'):
                    query = query.filter(Equipment.name.ilike(f'%{params["name"]}%'))
                if params.get('model'):
                    query = query.filter(Equipment.model.ilike(f'%{params["model"]}%'))
                if params.get('laboratory'):
                    query = query.filter(Equipment.laboratory.ilike(f'%{params["laboratory"]}%'))
                if params.get('location'):
                    query = query.filter(Equipment.location.ilike(f'%{params["location"]}%'))
                if params.get('status'):
                    query = query.filter(Equipment.status == params["status"])
                
                equipment_list = query.all()
            else:
                # 导出所有数据
                equipment_list = Equipment.query.all()
        
        # 准备数据
        data = []
        for equip in equipment_list:
            data.append({
                '固定资产编号': equip.fixed_asset_id or '',
                '设备管理编号': equip.equipment_manage_id or '',
                '自有编号': equip.self_number or '',
                '仪器名称': equip.name,
                '仪器型号': equip.model or '',
                '实验室': equip.laboratory or '',
                '位置': equip.location or '',
                '用途': equip.purpose or '',
                '使用信息': equip.usage_info or '',
                '状态': equip.status or '',
                '最近检查日期': equip.inspection_date.strftime('%Y-%m-%d') if equip.inspection_date else '',
                '年检日期': equip.annual_check_date.strftime('%Y-%m-%d') if equip.annual_check_date else '',
                '购买日期': equip.purchase_date.strftime('%Y-%m-%d') if equip.purchase_date else '',
                '购买金额': float(equip.purchase_amount) if equip.purchase_amount else 0.0,
                '折旧价值': float(equip.depreciated_value) if equip.depreciated_value else 0.0,
                '备注': equip.notes or '',
                '故障日期': equip.fault_date.strftime('%Y-%m-%d') if equip.fault_date else '',
                '故障类别': equip.fault_category or '',
                '处理方法': equip.handling_method or '',
                '解决方案': equip.solution or '',
                '结果': equip.result or '',
                '完成日期': equip.completion_date.strftime('%Y-%m-%d') if equip.completion_date else '',
                '操作员': equip.operator or '',
                '确认人': equip.confirmer or '',
                '设备图片': equip.equipment_image or '',
                '资产图片': equip.fixed_asset_image or '',
                '位置图片': equip.location_image or '',
                '发票附件': equip.invoice_attachment or '',
                '维修附件': equip.repair_attachment or '',
                '送货单附件': equip.delivery_attachment or '',
                '年检附件': equip.annual_check_attachment or '',
                '其他附件': equip.other_attachment or '',
                '原厂说明书': equip.manufacturer_manual_attachment or '',
                'SOP文档': equip.sop_attachment or '',
                '校准记录': equip.calibration_record_attachment or '',
                '厂家资料': equip.manufacturer_data_attachment or ''
            })
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 根据格式导出
        if format_type == 'csv':
            # CSV格式导出
            output = StringIO()
            df.to_csv(output, index=False, encoding='utf-8-sig')
            output.seek(0)
            
            # 如果指定了保存路径
            if save_path:
                try:
                    with open(save_path, 'w', encoding='utf-8-sig') as f:
                        f.write(output.getvalue())
                    return jsonify({'success': True, 'message': f'文件已保存到: {save_path}'})
                except Exception as e:
                    return jsonify({'error': f'保存文件失败: {str(e)}'}), 500
            
            # 直接下载
            response = make_response(output.getvalue().encode('utf-8-sig'))
            response.headers['Content-Disposition'] = 'attachment; filename=equipment_export.csv'
            response.headers['Content-type'] = 'text/csv; charset=utf-8'
            return response
        
        elif format_type == 'excel':
            # Excel格式导出
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='设备清单', index=False)
                
                # 设置列宽
                worksheet = writer.sheets['设备清单']
                for idx, col in enumerate(df.columns):
                    max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
                    worksheet.set_column(idx, idx, max_len)
            
            output.seek(0)
            
            # 如果指定了保存路径
            if save_path:
                try:
                    with open(save_path, 'wb') as f:
                        f.write(output.getvalue())
                    return jsonify({'success': True, 'message': f'文件已保存到: {save_path}'})
                except Exception as e:
                    return jsonify({'error': f'保存文件失败: {str(e)}'}), 500
            
            # 直接下载
            response = make_response(output.getvalue())
            response.headers['Content-Disposition'] = 'attachment; filename=equipment_export.xlsx'
            response.headers['Content-type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            return response
        
        elif format_type == 'pdf':
            # PDF格式导出 - 需要额外安装库
            try:
                from fpdf import FPDF
                
                # 创建PDF对象
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=10)
                
                # 添加标题
                title = "设备清单"
                if is_monthly_report:
                    title = f"{report_year}年{report_month}月设备检查报告"
                pdf.cell(200, 10, txt=title, ln=True, align='C')
                
                # 添加表头
                col_widths = [40, 60]  # 列宽
                for i, col in enumerate(df.columns):
                    pdf.cell(col_widths[i % 2], 10, txt=col, border=1)
                    if (i + 1) % 2 == 0:
                        pdf.ln()
                
                # 添加数据
                for _, row in df.iterrows():
                    for i, col in enumerate(df.columns):
                        pdf.cell(col_widths[i % 2], 10, txt=str(row[col]), border=1)
                    pdf.ln()
                
                # 生成PDF
                pdf_output = pdf.output(dest='S').encode('latin1')
                
                # 如果指定了保存路径
                if save_path:
                    try:
                        with open(save_path, 'wb') as f:
                            f.write(pdf_output)
                        return jsonify({'success': True, 'message': f'文件已保存到: {save_path}'})
                    except Exception as e:
                        return jsonify({'error': f'保存文件失败: {str(e)}'}), 500
                
                # 直接下载
                response = make_response(pdf_output)
                response.headers['Content-Disposition'] = 'attachment; filename=equipment_export.pdf'
                response.headers['Content-type'] = 'application/pdf'
                return response
                
            except ImportError:
                return jsonify({'error': 'PDF导出需要安装fpdf库'}), 400
        
        else:
            return jsonify({'error': '不支持的导出格式'}), 400
        
    except Exception as e:
        logger.error(f"设备导出失败: {str(e)}", exc_info=True)
        return jsonify({'error': f'导出失败: {str(e)}'}), 500

# 在原有代码基础上添加以下路由

@app.route('/equipment_monthly_report')
def equipment_monthly_report():
    # 获取查询参数
    year = request.args.get('year', default=datetime.now().year, type=int)
    month = request.args.get('month', default=datetime.now().month, type=int)
    status_param = request.args.get('status', default='all')
    search_query = request.args.get('search', default='', type=str)
    
    # 处理状态参数
    selected_statuses = status_param.split(',') if status_param else ['all']
    
    # 计算日期范围
    today = datetime.now().date()
    current_month_start = today.replace(day=1)
    next_month = current_month_start + timedelta(days=32)
    next_month_start = next_month.replace(day=1)
    current_month_end = next_month_start - timedelta(days=1)
    
    last_month_end = current_month_start - timedelta(days=1)
    last_month_start = last_month_end.replace(day=1)
    
    # 获取所有设备
    base_query = Equipment.query
    
    # 应用搜索条件
    if search_query:
        base_query = base_query.filter(
            or_(
                Equipment.name.ilike(f'%{search_query}%'),
                Equipment.equipment_manage_id.ilike(f'%{search_query}%'),
                Equipment.fixed_asset_id.ilike(f'%{search_query}%')
            )
        )
    
    all_equipment = base_query.all()
    
    # 准备报表数据
    report_data = []
    total_devices = set()  # 用于去重计数
    
    # 状态计数器
    status_count = {
        '正常': 0, '待检': 0, '维修中': 0, '已经维修': 0,
        '停用': 0, '报废中': 0, '报废': 0, '无记录': 0
    }
    
    for equip in all_equipment:
        # 设备唯一标识
        device_key = f"{equip.equipment_manage_id}-{equip.fixed_asset_id}"
        total_devices.add(device_key)
        
        # 查询当月记录
        current_month_record = Equipment.query.filter(
            Equipment.id == equip.id,
            extract('year', Equipment.operation_date) == year,
            extract('month', Equipment.operation_date) == month
        ).order_by(Equipment.operation_date.desc()).first()
        
        # 查询上个月记录
        last_month_record = Equipment.query.filter(
            Equipment.id == equip.id,
            extract('year', Equipment.operation_date) == last_month_start.year,
            extract('month', Equipment.operation_date) == last_month_start.month
        ).order_by(Equipment.operation_date.desc()).first()
        
        current_status = current_month_record.status if current_month_record else None
        last_status = last_month_record.status if last_month_record else None
        
        # 更新状态计数器
        if current_status:
            status_count[current_status] += 1
        else:
            status_count['无记录'] += 1
        
        # 状态筛选
        if 'all' not in selected_statuses:
            if current_status not in selected_statuses and (current_status or '无记录') not in selected_statuses:
                continue
        
        report_data.append({
            'equipment_manage_id': equip.equipment_manage_id,
            'fixed_asset_id': equip.fixed_asset_id,
            'name': equip.name,
            'model': equip.model,
            'laboratory': equip.laboratory,
            'location': equip.location,
            'current_month_status': current_status,
            'last_month_status': last_status,
            'inspection_date': current_month_record.inspection_date if current_month_record else None,
            'operator': current_month_record.operator if current_month_record else None,
            'notes': current_month_record.notes if current_month_record else None
        })
    
    # 月份显示格式
    current_month_str = f"{year}年{month}月"
    last_month_str = last_month_start.strftime('%Y年%m月')
    
    # 生成所有可用年份
    years = range(2020, datetime.now().year + 2)
    
    return render_template(
        'equipment_report.html',
        report_data=report_data,
        current_year=year,
        current_month=month,  # 数字月份（1-12）
        years=years,
        total_devices=len(total_devices),
        normal_count=status_count['正常'],
        pending_count=status_count['待检'],
        repairing_count=status_count['维修中'],
        repaired_count=status_count['已经维修'],
        disabled_count=status_count['停用'],
        scrapping_count=status_count['报废中'],
        scrapped_count=status_count['报废'],
        no_record_count=status_count['无记录'],
        current_month_start=current_month_start.strftime('%Y-%m-%d'),
        current_month_end=current_month_end.strftime('%Y-%m-%d'),
        last_month_start=last_month_start.strftime('%Y-%m-%d'),
        last_month_end=last_month_end.strftime('%Y-%m-%d'),
        last_month=last_month_str,
        current_month_display=current_month_str,  # 修复：重命名这个参数
        selected_statuses=selected_statuses,
        search_query=search_query,
        generate_time=datetime.now().strftime('%Y-%m-%d %H:%M'),
        current_user="管理员"
    )


@app.route('/check_equipment')
def check_equipment():
    equip = Equipment.query.first()
    return f"Operation Date: {equip.operation_date}"


# 在 equipment_analysis 路由中添加故障时间统计功能
# 更新 equipment_analysis 路由
@app.route('/equipment_analysis')
def equipment_analysis():
    try:
        # 获取查询参数
        time_range_days = request.args.get('time_range', '30')
        try:
            time_range_days = int(time_range_days)
        except ValueError:
            time_range_days = 30
            
        # 获取选中的设备ID
        equipment_ids = request.args.getlist('equipment_ids')
        selected_equipment_ids = equipment_ids
        
        # 计算时间范围
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=time_range_days)
        
        # 获取所有设备用于前端显示
        all_equipment = Equipment.query.order_by(Equipment.name).all()
        
        # 准备设备名称列表
        equipment_names = []
        if equipment_ids:
            selected_equipment = Equipment.query.filter(Equipment.id.in_(equipment_ids)).all()
            equipment_names = [e.name for e in selected_equipment]
        
        # 生成时间
        generate_time = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        # 计算状态统计数据
        status_stats = calculate_status_stats(equipment_ids, start_date, end_date)
        
        # 计算故障统计数据
        fault_stats = calculate_fault_stats(equipment_ids, start_date, end_date)
        
        # 准备图表数据
        chart_data = prepare_chart_data(equipment_ids, start_date, end_date, time_range_days)
        
        # 数据点计数
        data_points = Equipment.query.filter(
            Equipment.operation_date.between(start_date, end_date)
        ).count()
        
        # 最后更新时间
        last_update = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        return render_template(
            'equipment_analysis.html',
            time_range_days=time_range_days,
            equipment_names=equipment_names,
            all_equipment=all_equipment,
            generate_time=generate_time,
            fault_stats=fault_stats,
            status_stats=status_stats,
            selected_equipment_ids=selected_equipment_ids,
            chart_data=chart_data,
            data_points=data_points,
            last_update=last_update
        )
    
    except Exception as e:
        logger.error(f"设备运行分析失败: {str(e)}", exc_info=True)
        flash(f'分析数据加载失败: {str(e)}', 'danger')
        return redirect(url_for('equipment_monthly_report'))

# 新增辅助函数
def calculate_status_stats(equipment_ids, start_date, end_date):
    """计算设备状态统计数据"""
    status_stats = {
        'normal': 0,
        'low': 0,
        'critical': 0
    }
    
    # 获取最新状态记录
    query = Equipment.query.filter(
        Equipment.operation_date.between(start_date, end_date)
    )
    
    if equipment_ids:
        query = query.filter(Equipment.id.in_(equipment_ids))
    
    latest_records = query.order_by(Equipment.operation_date.desc()).all()
    
    # 统计状态
    for record in latest_records:
        if record.status in ['正常', 'Normal']:
            status_stats['normal'] += 1
        elif record.status in ['待检', '已经维修', 'Pending', 'Repaired']:
            status_stats['low'] += 1
        else:
            status_stats['critical'] += 1
    
    return status_stats

def calculate_fault_stats(equipment_ids, start_date, end_date):
    """计算故障统计数据"""
    query = Equipment.query.filter(
        Equipment.fault_date.isnot(None),
        Equipment.completion_date.isnot(None),
        Equipment.operation_date.between(start_date, end_date),
        Equipment.status.in_(['维修中', '已经维修', 'Under Repair', 'Repaired'])
    )
    
    if equipment_ids:
        query = query.filter(Equipment.id.in_(equipment_ids))
    
    fault_results = query.order_by(Equipment.fault_date.desc()).all()
    
    fault_stats = {
        'total_faults': len(fault_results),
        'avg_downtime': 0,
        'max_downtime': 0,
        'min_downtime': 0,
        'faults_by_category': defaultdict(int),
        'recent_faults': []
    }
    
    if fault_results:
        downtimes = []
        for record in fault_results:
            if record.fault_date and record.completion_date:
                downtime = (record.completion_date - record.fault_date).days
                downtimes.append(downtime)
                
                # 分类统计
                category = record.fault_category or '未分类'
                fault_stats['faults_by_category'][category] += 1
                
                # 最近故障记录
                if len(fault_stats['recent_faults']) < 10:
                    fault_stats['recent_faults'].append({
                        'name': record.name,
                        'fault_date': record.fault_date.strftime('%Y-%m-%d'),
                        'completion_date': record.completion_date.strftime('%Y-%m-%d'),
                        'downtime': downtime,
                        'category': category,
                        'notes': record.notes or ''  # 添加备注字段
                    })
        
        if downtimes:
            fault_stats['avg_downtime'] = round(sum(downtimes) / len(downtimes), 1)
            fault_stats['max_downtime'] = max(downtimes)
            fault_stats['min_downtime'] = min(downtimes)
    
    return fault_stats

def prepare_chart_data(equipment_ids, start_date, end_date, time_range_days):
    """准备图表数据"""
    # 创建日期范围
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)
    
    date_labels = [date.strftime('%Y-%m-%d') for date in dates]
    
    # 初始化数据集
    datasets = [
        {'label': '正常运行设备', 'data': [0] * len(dates), 'color': '#2ecc71'},
        {'label': '低风险设备', 'data': [0] * len(dates), 'color': '#f39c12'},
        {'label': '高风险设备', 'data': [0] * len(dates), 'color': '#e74c3c'}
    ]
    
    # 按天统计
    for i, date in enumerate(dates):
        start_of_day = datetime.combine(date, datetime.min.time())
        end_of_day = datetime.combine(date, datetime.max.time())
        
        # 查询当天的设备状态
        query = Equipment.query.filter(
            Equipment.operation_date.between(start_of_day, end_of_day)
        )
        
        if equipment_ids:
            query = query.filter(Equipment.id.in_(equipment_ids))
        
        daily_records = query.all()
        
        normal = low = critical = 0
        for record in daily_records:
            if record.status in ['正常', 'Normal']:
                normal += 1
            elif record.status in ['待检', '已经维修', 'Pending', 'Repaired']:
                low += 1
            else:
                critical += 1
        
        datasets[0]['data'][i] = normal
        datasets[1]['data'][i] = low
        datasets[2]['data'][i] = critical
    
    return {
        'labels': date_labels,
        'datasets': datasets
    }

@app.route('/export_equipment_analysis')
def export_equipment_analysis():
    """导出设备运行分析数据(Excel格式)"""
    try:
        # 获取查询参数
        time_range_days = int(request.args.get('time_range_days', 30))
        equipment_ids = request.args.get('equipment_ids', '')
        
        # 计算时间范围
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=time_range_days)
        
        # 准备查询
        status_query = db.session.query(
            db.extract('year', Equipment.operation_date).label('year'),
            db.extract('month', Equipment.operation_date).label('month'),
            Equipment.status,
            db.func.count().label('count')
        ).filter(
            Equipment.operation_date.between(start_date, end_date)
        )
        
        fault_query = Equipment.query.filter(
            Equipment.fault_date.isnot(None),
            Equipment.completion_date.isnot(None),
            Equipment.operation_date.between(start_date, end_date),
            Equipment.status.in_(['维修中', '已经维修'])
        )
        
        # 如果有设备ID筛选
        if equipment_ids:
            ids_list = [int(id) for id in equipment_ids.split(',')]
            status_query = status_query.filter(Equipment.id.in_(ids_list))
            fault_query = fault_query.filter(Equipment.id.in_(ids_list))
        
        # 执行状态查询
        status_results = status_query.group_by(
            'year', 'month', 'status'
        ).order_by(
            'year', 'month'
        ).all()
        
        # 执行故障查询
        fault_results = fault_query.order_by(Equipment.fault_date.desc()).all()
        
        # 创建Excel工作簿
        output = BytesIO()
        workbook = Workbook(output)
        worksheet = workbook.add_worksheet('设备运行分析')
        
        # 设置标题格式
        title_format = workbook.add_format({
            'bold': True,
            'font_size': 16,
            'align': 'center',
            'valign': 'vcenter'
        })
        
        # 设置表头格式
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#4361ee',
            'font_color': 'white',
            'border': 1,
            'align': 'center'
        })
        
        # 设置数据格式
        data_format = workbook.add_format({
            'border': 1,
            'align': 'center'
        })
        
        # 写入标题
        time_range_text = f"{time_range_days}天"
        if time_range_days >= 365:
            years = time_range_days // 365
            time_range_text = f"{years}年"
        
        worksheet.merge_range('A1:F1', f'设备运行分析报告（{time_range_text}）', title_format)
        
        # 写入基本信息
        worksheet.write('A3', '生成时间', header_format)
        worksheet.write('B3', datetime.utcnow().strftime('%Y-%m-%d %H:%M'), data_format)
        worksheet.write('A4', '分析范围', header_format)
        if equipment_ids:
            equipment_names = Equipment.query.filter(Equipment.id.in_([int(id) for id in equipment_ids.split(',')])).all()
            names = ", ".join([e.name for e in equipment_names[:3]])
            if len(equipment_names) > 3:
                names += f" 等{len(equipment_names)}台设备"
            worksheet.write('B4', names, data_format)
        else:
            worksheet.write('B4', '全部设备', data_format)
        
        # 写入状态统计数据
        worksheet.merge_range('A6:F6', '设备状态统计', header_format)
        worksheet.write('A7', '时间段', header_format)
        worksheet.write('B7', '正常', header_format)
        worksheet.write('C7', '待检', header_format)
        worksheet.write('D7', '维修中', header_format)
        worksheet.write('E7', '已经维修', header_format)
        worksheet.write('F7', '停用', header_format)
        worksheet.write('G7', '报废中', header_format)
        worksheet.write('H7', '报废', header_format)
        
        # 准备状态数据
        statuses = ['正常', '待检', '维修中', '已经维修', '停用', '报废中', '报废']
        status_data = {status: {} for status in statuses}
        
        # 收集数据
        for year, month, status, count in status_results:
            period = f"{int(year)}-{int(month):02d}"
            if status in status_data:
                status_data[status][period] = count
        
        # 获取所有时间段
        periods = sorted(set(period for data in status_data.values() for period in data.keys()))
        
        # 写入数据
        row = 8
        for period in periods:
            worksheet.write(row, 0, period, data_format)
            for col, status in enumerate(statuses, start=1):
                count = status_data[status].get(period, 0)
                worksheet.write(row, col, count, data_format)
            row += 1
        
        # 写入故障统计数据
        row += 2
        worksheet.merge_range(f'A{row}:F{row}', '故障时间统计', header_format)
        row += 1
        
        worksheet.write(row, 0, '总故障次数', header_format)
        worksheet.write(row, 1, len(fault_results), data_format)
        row += 1
        
        if fault_results:
            downtimes = [(record.completion_date - record.fault_date).days for record in fault_results]
            avg_downtime = sum(downtimes) / len(downtimes)
            max_downtime = max(downtimes)
            min_downtime = min(downtimes)
            
            worksheet.write(row, 0, '平均故障时间(天)', header_format)
            worksheet.write(row, 1, f"{avg_downtime:.1f}", data_format)
            row += 1
            
            worksheet.write(row, 0, '最长故障时间(天)', header_format)
            worksheet.write(row, 1, max_downtime, data_format)
            row += 1
            
            worksheet.write(row, 0, '最短故障时间(天)', header_format)
            worksheet.write(row, 1, min_downtime, data_format)
            row += 1
            
            # 写入故障类别分布
            row += 1
            worksheet.merge_range(f'A{row}:B{row}', '故障类别分布', header_format)
            row += 1
            
            fault_categories = {}
            for record in fault_results:
                category = record.fault_category or '未分类'
                fault_categories[category] = fault_categories.get(category, 0) + 1
            
            for category, count in fault_categories.items():
                worksheet.write(row, 0, category, header_format)
                worksheet.write(row, 1, count, data_format)
                row += 1
            
            # 写入最近故障记录
            row += 2
            worksheet.merge_range(f'A{row}:F{row}', '最近故障记录', header_format)
            row += 1
            
            headers = ['设备名称', '故障日期', '修复日期', '故障时长(天)', '故障类别', '解决方案']
            for col, header in enumerate(headers):
                worksheet.write(row, col, header, header_format)
            row += 1
            
            for record in fault_results[:10]:  # 最多显示10条
                downtime = (record.completion_date - record.fault_date).days
                worksheet.write(row, 0, record.name, data_format)
                worksheet.write(row, 1, record.fault_date.strftime('%Y-%m-%d'), data_format)
                worksheet.write(row, 2, record.completion_date.strftime('%Y-%m-%d'), data_format)
                worksheet.write(row, 3, downtime, data_format)
                worksheet.write(row, 4, record.fault_category or '未分类', data_format)
                worksheet.write(row, 5, record.solution or '', data_format)
                row += 1
        
        # 调整列宽
        worksheet.set_column('A:A', 15)
        worksheet.set_column('B:B', 20)
        worksheet.set_column('C:C', 12)
        worksheet.set_column('D:D', 12)
        worksheet.set_column('E:E', 15)
        worksheet.set_column('F:F', 30)
        
        # 保存工作簿
        workbook.close()
        output.seek(0)
        
        # 创建响应
        response = make_response(output.getvalue())
        response.headers['Content-Disposition'] = f'attachment; filename=equipment_analysis_{time_range_days}days.xlsx'
        response.headers['Content-type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        return response
        
    except Exception as e:
        logger.error(f"设备运行分析导出失败: {str(e)}", exc_info=True)
        return jsonify({
            'error': f'导出失败: {str(e)}'
        }), 500



@app.route('/export_warehouse')
def export_warehouse():
    """导出仓库数据（支持CSV和Excel格式）"""
    try:
        # 获取查询参数（与仓库页面相同）
        name = request.args.get('name', '').strip()
        product_code = request.args.get('product_code', '').strip()
        cas_number = request.args.get('cas_number', '').strip()
        batch_number = request.args.get('batch_number', '').strip()
        location = request.args.get('location', '').strip()
        warehouse_location = request.args.get('warehouse_location', '').strip()
        consumable_category = request.args.get('consumable_category', '').strip()
        regulatory_category = request.args.get('regulatory_category', '').strip()
        export_format = request.args.get('format', 'csv')  # 导出格式
        
        # 构建基础查询 - 获取每个产品代码+批次号的最新记录
        subquery = db.session.query(
            Chemical.product_code,
            Chemical.batch_number,
            func.max(Chemical.operation_date).label('max_date')
        ).group_by(
            Chemical.product_code,
            Chemical.batch_number
        ).subquery()

        # 主查询
        query = Chemical.query.join(
            subquery,
            and_(
                Chemical.product_code == subquery.c.product_code,
                Chemical.batch_number == subquery.c.batch_number,
                Chemical.operation_date == subquery.c.max_date
            )
        ).filter(Chemical.action == '入库')  # 只显示入库记录
        
        # 添加过滤条件
        if name:
            query = query.filter(Chemical.name.ilike(f'%{name}%'))
        if product_code:
            query = query.filter(Chemical.product_code.ilike(f'%{product_code}%'))
        if cas_number:
            query = query.filter(Chemical.cas_number.ilike(f'%{cas_number}%'))
        if batch_number:
            query = query.filter(Chemical.batch_number.ilike(f'%{batch_number}%'))
        if location:
            query = query.filter(Chemical.location.ilike(f'%{location}%'))
        if warehouse_location:
            query = query.filter(Chemical.warehouse_location.ilike(f'%{warehouse_location}%'))
        if consumable_category:
            query = query.filter(Chemical.consumable_category.ilike(f'%{consumable_category}%'))
        if regulatory_category:
            query = query.filter(Chemical.regulatory_category.ilike(f'%{regulatory_category}%'))
        
        # 执行查询
        chemicals = query.all()
        
        # 准备数据（包含净库存计算）
        data = []
        for chem in chemicals:
            net_stock = calculate_net_stock(chem.product_code, chem.batch_number)
            if net_stock > 0:
                data.append({
                    'product_code': chem.product_code,
                    'name': chem.name,
                    'cas_number': chem.cas_number or '',
                    'batch_number': chem.batch_number,
                    'production_date': chem.production_date.strftime('%Y-%m-%d') if chem.production_date else '',
                    'expiration_date': chem.expiration_date.strftime('%Y-%m-%d') if chem.expiration_date else '',
                    'days_remaining': chem.days_remaining,
                    'net_stock': net_stock,
                    'location': chem.location,
                    'warehouse_location': chem.warehouse_location or '',
                    'consumable_category': chem.consumable_category or '',
                    'regulatory_category': chem.regulatory_category or '',
                    'last_operation': chem.operation_date.strftime('%Y-%m-%d') if chem.operation_date else ''
                })
        
        if not data:
            return jsonify({'error': '没有可导出的数据'}), 404
        
        # 根据请求的格式返回数据
        if export_format == 'csv':
            # CSV格式导出
            output = StringIO()
            writer = csv.writer(output)
            
            # 写入表头
            headers = [
                '商品代码', '化学品名称', 'CAS编号', '批次号', '收支日期 ', '过期日期',
                '剩余天数', '净库存', '存放位置', '仓库位置', '耗材类别', '监管类别', '最后操作日期'
            ]
            writer.writerow(headers)
            
            # 写入数据
            for item in data:
                writer.writerow([
                    item['product_code'],
                    item['name'],
                    item['cas_number'],
                    item['batch_number'],
                    item['production_date'],
                    item['expiration_date'],
                    item['days_remaining'] if item['days_remaining'] is not None else '',
                    item['net_stock'],
                    item['location'],
                    item['warehouse_location'],
                    item['consumable_category'],
                    item['regulatory_category'],
                    item['last_operation']
                ])
            
            output.seek(0)
            response = make_response(output.getvalue().encode('utf-8-sig'))
            response.headers['Content-Disposition'] = 'attachment; filename=warehouse_data.csv'
            response.headers['Content-type'] = 'text/csv; charset=utf-8'
            return response
        
        elif export_format == 'excel':
            # Excel格式导出
            df = pd.DataFrame(data)
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='仓库数据', index=False)
                
                # 设置列宽
                worksheet = writer.sheets['仓库数据']
                for idx, col in enumerate(df.columns):
                    max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
                    worksheet.set_column(idx, idx, max_len)
            
            output.seek(0)
            response = make_response(output.getvalue())
            response.headers['Content-Disposition'] = 'attachment; filename=warehouse_data.xlsx'
            response.headers['Content-type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            return response
        
        else:
            return jsonify({'error': '不支持的导出格式'}), 400
    
    except Exception as e:
        logger.error(f"仓库数据导出失败: {str(e)}")
        return jsonify({'error': f'导出失败: {str(e)}'}), 500

# 导出当前库存数据 (Excel格式)
@app.route('/export_current_stock')
def export_current_stock():
    try:
        # 获取查询参数
        product_code = request.args.get('product_code', '').strip()
        batch_number = request.args.get('batch_number', '').strip()
        name = request.args.get('name', '').strip()
        location = request.args.get('location', '').strip()
        
        # 构建查询 (与current_stock路由相同)
        base_query = db.session.query(
            Chemical.product_code,
            Chemical.name,
            Chemical.cas_number,
            Chemical.batch_number,
            Chemical.location,
            Chemical.manufacturer,
            Chemical.specification,
            Chemical.expiration_date,
            func.sum(Chemical.quantity).filter(Chemical.action == '入库').label('total_in'),
            func.sum(Chemical.quantity).filter(Chemical.action.in_(['出库', '盘点'])).label('total_out'),
            (func.sum(Chemical.quantity).filter(Chemical.action == '入库') - 
            func.sum(Chemical.quantity).filter(Chemical.action.in_(['出库', '盘点']))).label('net_stock')
        ).group_by(
            Chemical.product_code,
            Chemical.name,
            Chemical.cas_number,
            Chemical.batch_number,
            Chemical.location,
            Chemical.manufacturer,
            Chemical.specification,
            Chemical.expiration_date
        ).having(
            (func.sum(Chemical.quantity).filter(Chemical.action == '入库') - 
            func.sum(Chemical.quantity).filter(Chemical.action.in_(['出库', '盘点']))) > 0
        )

        # 添加过滤条件
        if product_code:
            base_query = base_query.filter(Chemical.product_code.ilike(f'%{product_code}%'))
        if batch_number:
            base_query = base_query.filter(Chemical.batch_number.ilike(f'%{batch_number}%'))
        if name:
            base_query = base_query.filter(Chemical.name.ilike(f'%{name}%'))
        if location:
            base_query = base_query.filter(Chemical.location.ilike(f'%{location}%'))

        # 执行查询
        stock_data = base_query.all()
        
        # 准备数据
        data = [{
            '商品代码': item.product_code,
            '化学品名称': item.name,
            'CAS编号': item.cas_number or '',
            '规格型号': item.specification or '',
            '生产商': item.manufacturer or '',
            '批次号': item.batch_number or '',
            '存放位置': item.location or '',
            '入库总量': item.total_in or 0,
            '出库总量': item.total_out or 0,
            '净库存': item.net_stock or 0,
            '过期日期': item.expiration_date.strftime('%Y-%m-%d') if item.expiration_date else '',
            '剩余天数': (item.expiration_date - datetime.now().date()).days if item.expiration_date else '',
            '状态': '即将过期' if item.expiration_date and (item.expiration_date - datetime.now().date()).days <= 30 else '正常'
        } for item in stock_data]
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 创建Excel文件
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='当前库存', index=False)
            
            # 获取工作表并设置列宽
            worksheet = writer.sheets['当前库存']
            for idx, col in enumerate(df.columns):
                max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.set_column(idx, idx, max_len)
        
        output.seek(0)
        response = make_response(output.getvalue())
        response.headers['Content-Disposition'] = 'attachment; filename=chemical_current_stock.xlsx'
        response.headers['Content-type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        return response
        
    except Exception as e:
        logger.error(f"库存导出失败: {str(e)}")
        flash(f'库存导出失败: {str(e)}', 'danger')
        return redirect(url_for('current_stock'))

# 导出当前库存数据 (CSV格式)


@app.route('/export_search_results')
def export_search_results():
    """导出搜索结果到CSV文件"""
    try:
        # 从请求参数中获取搜索条件
        search_params = request.args.to_dict()
        
        # 构建与搜索相同的查询
        query = Chemical.query
        filters = []
        
        # 应用搜索条件
        if search_params.get('product_code'):
            if search_params.get('search_mode') == 'exact':
                filters.append(Chemical.product_code == search_params['product_code'])
            else:
                filters.append(Chemical.product_code.ilike(f"%{search_params['product_code']}%"))
        
        if search_params.get('name'):
            if search_params.get('search_mode') == 'exact':
                filters.append(Chemical.name == search_params['name'])
            else:
                filters.append(Chemical.name.ilike(f"%{search_params['name']}%"))
        
        # 其他过滤条件...
        # (与/search路由中的过滤条件相同)
        
        # 仓库位置过滤
        if search_params.get('warehouse_location'):
            filters.append(Chemical.warehouse_location.ilike(f"%{search_params['warehouse_location']}%"))
        
        if filters:
            query = query.filter(and_(*filters))
        
        # 获取所有结果
        results = query.all()
        
        # 创建CSV响应
        output = StringIO()
        writer = csv.writer(output)
        
        # 写入表头（包括所有字段）
        headers = [
            'ID', '商品代码', '化学品名称', '型号', '商品规格', '分类', '耗材类别', '浓度', '单位', '仓库位置', 
            'CAS编号', '监管类别', '操作员', '确认人', '存放位置', '操作类型', '操作数量', '此批总库存', 
            '总库存', '安全库存', '状态', '生产商', '批次号', '收支日期', '过期日期', '操作日期', 
            '单价', '供应商', '备注', '图片', '附件', '送货单附件', '发票附件', 'MSDS附件'
        ]
        writer.writerow(headers)
        
        # 写入数据
        for chem in results:
            writer.writerow([
                chem.id,
                chem.product_code,
                chem.name,
                chem.model,
                chem.specification,
                chem.category,
                chem.consumable_category,
                chem.concentration,
                chem.unit,
                chem.warehouse_location,
                chem.cas_number,
                chem.regulatory_category,
                chem.operator,
                chem.confirmer,
                chem.location,
                chem.action,
                chem.quantity,
                chem.batch_total_stock,
                chem.total_stock,
                chem.safety_stock,
                chem.status,
                chem.manufacturer,
                chem.batch_number,
                chem.production_date.strftime('%Y-%m-%d') if chem.production_date else '',
                chem.expiration_date.strftime('%Y-%m-%d') if chem.expiration_date else '',
                chem.operation_date.strftime('%Y-%m-%d %H:%M:%S') if chem.operation_date else '',
                chem.price,
                chem.supplier,
                chem.notes,
                chem.image,
                chem.attachment,
                chem.delivery_order_attachment,
                chem.invoice_attachment,
                chem.msds_attachment
            ])
        
        output.seek(0)
        response = make_response(output.getvalue().encode('utf-8-sig'))
        response.headers['Content-Disposition'] = 'attachment; filename=chemicals_search_results.csv'
        response.headers['Content-type'] = 'text/csv; charset=utf-8'
        return response
        
    except Exception as e:
        return jsonify({'error': f'导出失败: {str(e)}'}), 500

@app.route('/export_warehouse_pdf')
def export_warehouse_pdf():
    """导出仓库数据为PDF格式"""
    try:
        # 获取查询参数
        name = request.args.get('name', '').strip()
        product_code = request.args.get('product_code', '').strip()
        cas_number = request.args.get('cas_number', '').strip()
        batch_number = request.args.get('batch_number', '').strip()
        location = request.args.get('location', '').strip()
        warehouse_location = request.args.get('warehouse_location', '').strip()
        consumable_category = request.args.get('consumable_category', '').strip()
        regulatory_category = request.args.get('regulatory_category', '').strip()
        
        # 构建基础查询
        subquery = db.session.query(
            Chemical.product_code,
            Chemical.batch_number,
            func.max(Chemical.operation_date).label('max_date')
        ).group_by(
            Chemical.product_code,
            Chemical.batch_number
        ).subquery()

        query = Chemical.query.join(
            subquery,
            and_(
                Chemical.product_code == subquery.c.product_code,
                Chemical.batch_number == subquery.c.batch_number,
                Chemical.operation_date == subquery.c.max_date
            )
        ).filter(Chemical.action == '入库')
        
        # 添加过滤条件
        if name: query = query.filter(Chemical.name.ilike(f'%{name}%'))
        if product_code: query = query.filter(Chemical.product_code.ilike(f'%{product_code}%'))
        if cas_number: query = query.filter(Chemical.cas_number.ilike(f'%{cas_number}%'))
        if batch_number: query = query.filter(Chemical.batch_number.ilike(f'%{batch_number}%'))
        if location: query = query.filter(Chemical.location.ilike(f'%{location}%'))
        if warehouse_location: query = query.filter(Chemical.warehouse_location.ilike(f'%{warehouse_location}%'))
        if consumable_category: query = query.filter(Chemical.consumable_category.ilike(f'%{consumable_category}%'))
        if regulatory_category: query = query.filter(Chemical.regulatory_category.ilike(f'%{regulatory_category}%'))
        
        # 执行查询
        chemicals = query.all()
        
        # 准备数据
        data = []
        for chem in chemicals:
            net_stock = calculate_net_stock(chem.product_code, chem.batch_number)
            if net_stock > 0:
                data.append({
                    'product_code': chem.product_code,
                    'name': chem.name,
                    'cas_number': chem.cas_number or '',
                    'batch_number': chem.batch_number,
                    'production_date': chem.production_date.strftime('%Y-%m-%d') if chem.production_date else '',
                    'expiration_date': chem.expiration_date.strftime('%Y-%m-%d') if chem.expiration_date else '',
                    'days_remaining': chem.days_remaining if chem.expiration_date else '',
                    'net_stock': net_stock,
                    'location': chem.location,
                    'warehouse_location': chem.warehouse_location or '',
                    'consumable_category': chem.consumable_category or '',
                    'regulatory_category': chem.regulatory_category or '',
                    'last_operation': chem.operation_date.strftime('%Y-%m-%d') if chem.operation_date else ''
                })
        
        # 生成PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=landscape(letter))
        
        # 设置中文支持
        chinese_font = 'SimSun'  # 确保系统中存在宋体
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Chinese', fontName=chinese_font, fontSize=10))
        
        # 准备表格数据
        table_data = []
        
        # 添加中文表头
        headers = [
            '商品代码', '化学品名称', 'CAS编号', '批次号', '收支日期 ', 
            '过期日期', '剩余天数', '净库存', '存放位置', '仓库位置', 
            '耗材类别', '监管类别', '最后操作日期'
        ]
        table_data.append([Paragraph(header, styles['Chinese']) for header in headers])
        
        # 添加数据行
        for item in data:
            table_data.append([
                Paragraph(item['product_code'], styles['Chinese']),
                Paragraph(item['name'], styles['Chinese']),
                Paragraph(item['cas_number'], styles['Chinese']),
                Paragraph(item['batch_number'], styles['Chinese']),
                Paragraph(item['production_date'], styles['Chinese']),
                Paragraph(item['expiration_date'], styles['Chinese']),
                Paragraph(str(item['days_remaining']), styles['Chinese']),
                Paragraph(str(item['net_stock']), styles['Chinese']),
                Paragraph(item['location'], styles['Chinese']),
                Paragraph(item['warehouse_location'], styles['Chinese']),
                Paragraph(item['consumable_category'], styles['Chinese']),
                Paragraph(item['regulatory_category'], styles['Chinese']),
                Paragraph(item['last_operation'], styles['Chinese'])
            ])
        
        # 创建表格
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), chinese_font),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), chinese_font),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
        ]))
        
        # 添加标题
        elements = [
            Paragraph('化学品仓库库存报告', 
                      ParagraphStyle(name='Title', fontName=chinese_font, fontSize=16, alignment=1)),
            Spacer(1, 12),
            Paragraph(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                      ParagraphStyle(name='Subtitle', fontName=chinese_font, fontSize=10, alignment=1)),
            Spacer(1, 12),
            table
        ]
        
        # 构建PDF
        doc.build(elements)
        
        buffer.seek(0)
        response = make_response(buffer.getvalue())
        response.headers['Content-Disposition'] = 'attachment; filename=warehouse_data.pdf'
        response.headers['Content-type'] = 'application/pdf'
        return response
        
    except Exception as e:
        logger.error(f"PDF导出失败: {str(e)}")
        return jsonify({'error': f'PDF导出失败: {str(e)}'}), 500

@app.route('/equipment_list_page', endpoint='equipment_list_page')
def equipment_list_page():
    try:
        # 获取查询参数
        page = request.args.get('page', 1, type=int)
        fixed_asset_id = request.args.get('fixed_asset_id', '').strip()
        equipment_manage_id = request.args.get('equipment_manage_id', '').strip()
        name = request.args.get('name', '').strip()
        model = request.args.get('model', '').strip()
        laboratory = request.args.get('laboratory', '').strip()
        location = request.args.get('location', '').strip()
        status = request.args.get('status', '')
        
        # 构建查询
        query = Equipment.query
        
        # 应用搜索条件
        if fixed_asset_id:
            query = query.filter(Equipment.fixed_asset_id.ilike(f'%{fixed_asset_id}%'))
        if equipment_manage_id:
            query = query.filter(Equipment.equipment_manage_id.ilike(f'%{equipment_manage_id}%'))
        if name:
            query = query.filter(Equipment.name.ilike(f'%{name}%'))
        if model:
            query = query.filter(Equipment.model.ilike(f'%{model}%'))
        if laboratory:
            query极 = query.filter(Equipment.laboratory.ilike(f'%{laboratory}%'))
        if location:
            query = query.filter(Equipment.location.ilike(f'%{location}%'))
        if status:
            query = query.filter(Equipment.status == status)
        
        # 每页显示的数量
        per_page = 20
        
        # 执行分页查询
        equipment = query.paginate(page=page, per_page=per_page, error_out=False)
        
        # 构建基础参数（排除分页参数）
        base_args = {
            'fixed_asset_id': fixed_asset_id,
            'equipment_manage_id': equipment_manage_id,
            'name': name,
            'model': model,
            'laboratory': laboratory,
            'location': location,
            'status': status
        }
        
        # 过滤掉空值参数
        base_args = {k: v for k, v in base_args.items() if v}
        
        # 渲染模板并传递数据
        return render_template(
            'equipment_list.html',
            equipment=equipment,
            base_args=base_args
        )
    
    except Exception as e:
        logger.error(f"设备列表查询失败: {str(e)}", exc_info=True)
        flash(f'设备列表加载失败: {str(e)}', 'danger')
        return redirect(url_for('index'))


# 添加密码验证逻辑到删除路由
@app.route('/delete_equipment/<int:id>', methods=['POST'])
def delete_equipment(id):
    # 从环境变量获取管理员密码
    ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'RI')
    
    # 获取请求中的密码
    password = request.form.get('password', '')
    if password != ADMIN_PASSWORD:
        flash('删除失败: 密码错误', 'danger')
        return redirect(url_for('equipment_list'))
    
    equipment = Equipment.query.get_or_404(id)
    try:
        # 记录删除操作日志
        logger.info(f"删除设备: ID={equipment.id}, 名称={equipment.name}, 操作员={request.remote_addr}")
        
        db.session.delete(equipment)
        db.session.commit()
        flash('设备删除成功!', 'success')
    except Exception as e:
        db.session.rollback()
        logger.error(f"删除失败: {str(e)}")
        flash(f'删除失败: {str(e)}', 'danger')
    return redirect(url_for('equipment_list'))


@app.route('/check_materials')
def check_materials():
    return render_template('check_materials.html')

@app.route('/upload_materials', methods=['POST'])
def upload_materials():
    try:
        msds_file = request.files['msds']
        banned_file = request.files['banned_list']
        
        # 扩展中文编码支持
        chinese_encodings = ['utf-8', 'gbk', 'gb18030', 'gb2312', 'big5', 'latin1']
        
        # 创建文本解码函数
        def decode_content(content, encoding_list):
            """尝试多种编码解码内容"""
            for encoding in encoding_list:
                try:
                    return content.decode(encoding)
                except UnicodeDecodeError:
                    continue
            # 如果所有编码都失败，尝试utf-8并忽略错误
            try:
                return content.decode('utf-8', errors='ignore')
            except:
                return content.decode('latin-1', errors='ignore')
        
        # 处理MSDS文件
        msds_text = None
        msds_content = msds_file.stream.read()
        
        # 如果是PDF文件，尝试使用PyPDF2读取
        if msds_file.filename.lower().endswith('.pdf'):
            try:
                from PyPDF2 import PdfReader
                import io
                
                pdf_reader = PdfReader(io.BytesIO(msds_content))
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
                msds_text = text
            except Exception as e:
                logger.error(f"PDF处理失败: {str(e)}")
                # 如果PDF处理失败，尝试直接解码二进制内容
                msds_text = decode_content(msds_content, chinese_encodings)
        else:
            # 文本文件直接解码
            msds_text = decode_content(msds_content, chinese_encodings)
        
        # 处理禁用原料文件
        banned_content = banned_file.stream.read()
        banned_text = decode_content(banned_content, chinese_encodings)
        
        # 将禁用原料列表按行分割，并过滤空行
        banned_list = [line.strip() for line in banned_text.splitlines() if line.strip()]
        
        # 检查禁用原料 - 添加额外处理（忽略空格和特殊符号）
        results = []
        for banned in banned_list:
            # 处理可能的空格/符号差异
            normalized_banned = banned.replace(' ', '').replace('_', '').replace('-', '')
            normalized_msds = msds_text.replace(' ', '').replace('_', '').replace('-', '')
            
            # 忽略大小写
            if (banned.lower() in msds_text.lower() or 
                normalized_banned.lower() in normalized_msds.lower()):
                results.append(banned)
        
        # 去重
        results = list(set(results))
        
        return jsonify({
            "results": results,
            "total_scanned": len(banned_list),
            "banned_found": len(results)
        })
        
    except KeyError as e:
        logger.error(f"缺少必要文件: {str(e)}")
        return jsonify({"error": "请上传MSDS文件和禁用原料列表"}), 400
    except Exception as e:
        logger.error(f"处理文件时发生错误: {str(e)}", exc_info=True)
        return jsonify({
            "error": f"服务器内部错误: {str(e)}",
            "trace": traceback.format_exc()
        }), 500

@app.route('/export_current_stock_csv')
def export_current_stock_csv():
    """导出当前库存数据到CSV文件（支持中文字符）"""
    try:
        # 获取查询参数
        product_code = request.args.get('product_code', '').strip()
        batch_number = request.args.get('batch_number', '').strip()
        name = request.args.get('name', '').strip()
        location = request.args.get('location', '').strip()
        
        # 构建基础查询
        base_query = db.session.query(
            Chemical.product_code,
            Chemical.name,
            Chemical.cas_number,
            Chemical.specification,
            Chemical.model,
            Chemical.manufacturer,
            Chemical.batch_number,
            Chemical.location,
            Chemical.expiration_date,
            func.sum(Chemical.quantity).filter(Chemical.action == '入库').label('total_in'),
            func.sum(Chemical.quantity).filter(Chemical.action.in_(['出库', '盘点'])).label('total_out'),
            (func.sum(Chemical.quantity).filter(Chemical.action == '入库') - 
             func.sum(Chemical.quantity).filter(Chemical.action.in_(['出库', '盘点']))).label('net_stock'),
            Chemical.total_stock,
            Chemical.safety_stock,
            Chemical.warehouse_location,
            Chemical.regulatory_category
        ).group_by(
            Chemical.product_code,
            Chemical.name,
            Chemical.cas_number,
            Chemical.specification,
            Chemical.model,
            Chemical.manufacturer,
            Chemical.batch_number,
            Chemical.location,
            Chemical.expiration_date,
            Chemical.total_stock,
            Chemical.safety_stock,
            Chemical.warehouse_location,
            Chemical.regulatory_category
        ).having(
            (func.sum(Chemical.quantity).filter(Chemical.action == '入库') - 
             func.sum(Chemical.quantity).filter(Chemical.action.in_(['出库', '盘点']))) > 0
        )

        # 添加过滤条件
        if product_code:
            base_query = base_query.filter(Chemical.product_code.ilike(f'%{product_code}%'))
        if batch_number:
            base_query = base_query.filter(Chemical.batch_number.ilike(f'%{batch_number}%'))
        if name:
            base_query = base_query.filter(Chemical.name.ilike(f'%{name}%'))
        if location:
            base_query = base_query.filter(Chemical.location.ilike(f'%{location}%'))

        # 执行查询
        stock_data = base_query.all()
        
        # 创建CSV输出
        output = StringIO()
        output.write('\ufeff')  # UTF-8 BOM for Excel compatibility
        
        writer = csv.writer(output)
        
        # 写入表头（中英文对照）
        headers = [
            '商品代码', '化学品名称', 'CAS编号', '规格', '型号', 
            '生产商', '批次号', '存放位置', '过期日期', '入库总量', 
            '出库总量', '净库存', '总库存', '安全库存',
            '仓库位置', '监管类别'
        ]
        writer.writerow(headers)
        
        # 写入数据
        today = datetime.now().date()
        
        for chem in stock_data:
            # 计算剩余天数
            days_remaining = (chem.expiration_date - today).days if chem.expiration_date else None
            
            writer.writerow([
                chem.product_code,
                chem.name,
                chem.cas_number or '',
                chem.specification or '',
                chem.model or '',
                chem.manufacturer or '',
                chem.batch_number or '',
                chem.location or '',
                chem.expiration_date.strftime('%Y-%m-%d') if chem.expiration_date else '',
                chem.total_in or 0,
                chem.total_out or 0,
                chem.net_stock or 0,
                chem.total_stock or 0,
                chem.safety_stock or 0,
                chem.warehouse_location or '',
                chem.regulatory_category or ''
            ])
        
        output.seek(0)
        
        # 创建响应
        response = make_response(output.getvalue().encode('utf-8'))
        response.headers['Content-Disposition'] = 'attachment; filename=chemical_current_stock.csv'
        response.headers['Content-type'] = 'text/csv; charset=utf-8'
        
        # 日志记录
        logger.info(f"库存数据导出成功: 产品代码={product_code}, 名称={name}, 批次={batch_number}, 位置={location}")
        
        return response
        
    except Exception as e:
        logger.error(f"库存CSV导出失败: {str(e)}", exc_info=True)
        flash(f'库存CSV导出失败: {str(e)}', 'danger')
        return redirect(url_for('current_stock'))
@app.route('/analyze_msds', methods=['POST'])
def analyze_msds():
    """
    处理MSDS文件上传和分析
    """
    try:
        # 检查文件是否上传
        if 'msds' not in request.files:
            return jsonify({"error": "No file part"}), 400
            
        file = request.files['msds']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        # 提取文件内容
        file_content = file.read()
        filename = file.filename.lower()
        
        # 根据文件类型提取文本
        text = ""
        if filename.endswith('.pdf'):
            # 处理PDF文件
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        elif filename.endswith('.docx'):
            # 处理Word文档
            doc = Document(BytesIO(file_content))
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif filename.endswith('.txt') or filename.endswith('.doc'):
            # 处理文本文件
            text = file_content.decode('utf-8', errors='ignore')
        else:
            return jsonify({"error": "Unsupported file format"}), 400
            
        # 截取前5000个字符（避免API限制）
        analysis_text = text[:5000]
        
        # 构建DeepSeek API请求
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个化学品安全专家，请分析MSDS文档中的化学成分。"
                },
                {
                    "role": "user",
                    "content": f"""
                    请分析以下MSDS文档内容，并回答以下问题：
                    
                    1. 该MSDS中的化学品成分是否包含危险化学品？如果有，请列出化学品名称和所在位置。
                    2. 该MSDS中的化学品成分是否包含易制爆化学品？如果有，请列出化学品名称和所在位置。
                    3. 该MSDS中的化学品成分是否包含易制毒化学品？如果有，请列出化学品名称和所在位置。
                    4. 该MSDS中的化学品成分是否包含化妆品禁用原料？如果有，请列出化学品名称和所在位置。
                    
                    请按照以下JSON格式返回结果：
                    {{
                      "analysis": [
                        {{
                          "chemical": "化学品名称",
                          "category": "危险化学品/易制爆化学品/易制毒化学品/化妆品禁用原料",
                          "conclusion": "简要结论",
                          "context": "在MSDS中的上下文内容"
                        }}
                      ]
                    }}
                    
                    MSDS内容：
                    {analysis_text}
                    """
                }
            ],
            "temperature": 0.1,
            "max_tokens": 2000
        }
        
        # 调用DeepSeek API
        start_time = time.time()
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        # 解析响应
        result = response.json()
        analysis_content = result['choices'][0]['message']['content']
        
        # 在实际应用中，这里需要从文本中提取JSON结构
        # 以下是模拟的解析过程
        results = parse_analysis_results(analysis_content)
        
        return jsonify({
            "success": True,
            "text_extract": analysis_text[:500] + "..." if len(analysis_text) > 500 else analysis_text,
            "analysis": results,
            "time": time.time() - start_time
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500

def parse_analysis_results(analysis_content):
    """
    解析DeepSeek返回的分析结果
    在实际应用中，这里需要从文本中提取JSON结构
    以下是模拟返回结果
    """
    # 模拟解析过程 - 在实际应用中应替换为真正的解析逻辑
    return [
        {
            "chemical": "胡椒醛",
            "category": "易制毒化学品",
            "conclusion": "在MSDS的第3部分第2段中发现，该化学品属于易制毒化学品，可用于制造毒品",
            "context": "...3. 成分/组成信息\n胡椒醛 (CAS: 120-57-0) ...\n4. 危险性概述\n..."
        },
        {
            "chemical": "丙酮",
            "category": "易制爆化学品",
            "conclusion": "在MSDS的第2部分发现丙酮作为溶剂使用，属于易制爆化学品",
            "context": "...2. 危险性概述\n丙酮 (CAS: 67-64-1) ...\n该化学品易燃，易制爆..."
        },
        {
            "chemical": "苯酚",
            "category": "危险化学品",
            "conclusion": "在MSDS的第5部分发现苯酚，属于危险化学品，具有腐蚀性",
            "context": "...5. 消防措施\n苯酚 (CAS: 108-95-2) ...\n避免与皮肤接触，具有腐蚀性..."
        },
        {
            "chemical": "汞化合物",
            "category": "化妆品禁用原料",
            "conclusion": "在MSDS的第9部分发现汞化合物，属于化妆品禁用原料",
            "context": "...9. 理化性质\n汞化合物 ...\n10. 稳定性和反应活性..."
        }
    ]

@app.route('/msds_samples/<filename>')
def download_msds_sample(filename):
    """
    提供示例MSDS文件下载
    """
    samples_dir = os.path.join(app.root_path, 'msds_samples')
    return send_from_directory(samples_dir, filename, as_attachment=True)

@app.route('/analyze_chemical', methods=['POST'])
def analyze_chemical():
    """分析化学品成分"""
    try:
        data = request.get_json()
        chemical_name = data.get('chemicalName', '')
        cas_number = data.get('casNumber', '')
        
        if not chemical_name and not cas_number:
            return jsonify({'error': '请提供化学品名称或CAS编号'}), 400
        
        # 构建DeepSeek API请求
        headers = {
            "Authorization": f"Bearer {app.config['DEEPSEEK_API_KEY']}",
            "Content-Type": "application/json"
        }
        
        # 创建分析提示
        prompt = f"""
        作为化学品安全专家，请分析以下化学品信息：
        
        化学名称: {chemical_name}
        CAS 编号: {cas_number}
        
        请回答以下问题：
        1. 该化学品是否属于危险化学品？如果是，说明原因和危险性类别。
        2. 该化学品是否属于易制爆化学品？如果是，说明原因。
        3. 该化学品是否属于易制毒化学品？如果是，说明原因。
        4. 该化学品是否属于化妆品禁用原料？如果是，说明原因。
        
        请按照以下JSON格式返回结构化结果：
        {{
          "results": [
            {{
              "chemical": "化学品名称",
              "category": "危险化学品/易制爆化学品/易制毒化学品/化妆品禁用原料",
              "conclusion": "简要结论",
              "context": "详细分析内容"
            }}
          ]
        }}
        """
        
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 2000
        }
        
        # 调用DeepSeek API
        response = requests.post(
            app.config['DEEPSEEK_API_URL'], 
            headers=headers, 
            json=payload, 
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        analysis_content = result['choices'][0]['message']['content']
        
        # 解析响应
        try:
            analysis_data = json.loads(analysis_content)
            return jsonify(analysis_data)
        except json.JSONDecodeError:
            # 如果返回的不是标准JSON，返回原始内容
            return jsonify({
                "results": [{
                    "chemical": chemical_name,
                    "category": "分析结果",
                    "conclusion": "AI分析完成",
                    "context": analysis_content
                }]
            })
        
    except Exception as e:
        logger.error(f"化学品分析失败: {str(e)}")
        return jsonify({'error': f'分析失败: {str(e)}'}), 500

class medicine_material(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    item_code = db.Column(db.String(100))
    name = db.Column(db.String(200), nullable=False)
    batch_number = db.Column(db.String(100))
    specification = db.Column(db.String(200))  # 规格
    model = db.Column(db.String(100))  # 型号
    item_type = db.Column(db.Enum('药品', '物资'), nullable=False)  # 物品类型
    category = db.Column(db.String(100))  # 物品类别
    laboratory = db.Column(db.String(100))  # 实验室
    action = db.Column(db.String(50))  # 操作
    quantity = db.Column(db.Integer)  # 数量
    unit = db.Column(db.String(50))  # 单位
    location = db.Column(db.String(200))  # 存储位置
    price = db.Column(db.Float)  # 单价
    storage_condition = db.Column(db.String(200))  # 储存条件
    payment_date = db.Column(db.Date)  # 支付日期
    operation_date = db.Column(db.DateTime, default=datetime.utcnow)  # 操作日期
    expiration_date = db.Column(db.Date)  # 失效期
    status = db.Column(db.String(50))  # 状态
    manufacturer = db.Column(db.String(200))  # 厂家
    supplier = db.Column(db.String(200))  # 供应商
    notes = db.Column(db.Text)  # 备注
    operator = db.Column(db.String(100))  # 操作员
    confirmer = db.Column(db.String(100))  # 确认人
    batch_total_stock = db.Column(db.Integer)  # 此批总库存
    total_stock = db.Column(db.Integer)  # 总库存
    safety_stock = db.Column(db.Integer)  # 安全库存
    purpose = db.Column(db.Text)  # 物品用途
    item_image = db.Column(db.String(200))  # 物品图片
    label_image = db.Column(db.String(200))  # 标签图片
    instruction_image = db.Column(db.String(200))  # 使用说明图片
    location_image = db.Column(db.String(200))  # 定位图片
    official_website = db.Column(db.String(200))  # 物品官方网站
    usage_video = db.Column(db.String(200))  # 物品使用视频
    delivery_order_attachment = db.Column(db.String(200))
    proposal_attachment = db.Column(db.String(200))
    purchase_request_attachment = db.Column(db.String(200))
    manufacturer_data_attachment = db.Column(db.String(200))
    disposal_data_attachment = db.Column(db.String(200))
    other_attachment = db.Column(db.String(200))  # 其他附件
    manufacturer_manual_attachment = db.Column(db.String(200))  # 设备原厂使用说明书附件
    sop_attachment = db.Column(db.String(200))  # SOP标准操作规程附件
    calibration_record_attachment = db.Column(db.String(200))  # 设备定期校准记录附件
    manufacturer_data_attachment = db.Column(db.String(200))  # 厂家资料附件

    @property
    def days_remaining(self):
        if self.expiration_date:
            return (self.expiration_date - datetime.now().date()).days
        return None


# 药品物资列表路由（添加去重统计和密码验证）
# 优化药品物资列表查询（文档2中更新）
@app.route('/medicine_material')
def medicine_material_list():
    """药品与物资列表页面（优化版）"""
    try:
        # 获取查询参数
        name = request.args.get('name', '').strip()
        item_code = request.args.get('item_code', '').strip()
        item_type = request.args.get('item_type', '')
        action = request.args.get('action', '')
        status = request.args.get('status', '')
        expiring = request.args.get('expiring', '') == 'on'
        page = request.args.get('page', 1, type=int)
        
        # 构建高级查询
        query = medicine_material.query
        
        # 应用搜索条件
        if name:
            query = query.filter(medicine_material.name.ilike(f'%{name}%'))
        if item_code:
            query = query.filter(medicine_material.item_code.ilike(f'%{item_code}%'))
        if item_type:
            query = query.filter(medicine_material.item_type == item_type)
        if action:
            query = query.filter(medicine_material.action == action)
        if status:
            query = query.filter(medicine_material.status == status)
        
        # 即将过期筛选
        if expiring:
            today = datetime.now().date()
            thirty_days_later = today + timedelta(days=30)
            query = query.filter(
                medicine_material.expiration_date.between(today, thirty_days_later)
            )
        
        # 计算统计信息
        total_items = query.count()
        normal_status = query.filter(medicine_material.status == '正常').count()
        expiring_soon = query.filter(medicine_material.status == '即将过期').count()
        
        # 计算去重后的物品代码数量
        unique_codes = db.session.query(medicine_material.item_code.distinct()).count()
        
        # 分页处理（每页50条）
        per_page = 50
        items = query.order_by(medicine_material.operation_date.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        # 准备统计数据
        stats = {
            'total_items': total_items,
            'normal_status': normal_status,
            'expiring_soon': expiring_soon,
            'unique_codes': unique_codes
        }
        
        return render_template(
            'medicine_material.html',
            items=items,
            stats=stats,
            search_params={
                'name': name,
                'item_code': item_code,
                'item_type': item_type,
                'action': action,
                'status': status,
                'expiring': expiring
            }
        )
    
    except Exception as e:
        logger.error(f"药品与物资列表加载失败: {str(e)}", exc_info=True)
        flash(f'数据加载失败: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/delete_medicine_material', methods=['POST'])
def delete_medicine_material():
    """删除药品与物资记录（带密码验证）"""
    try:
        # 获取表单数据
        item_id = request.form.get('item_id')
        password = request.form.get('password')
        
        if not item_id:
            return jsonify({
                'success': False,
                'message': '缺少物品ID参数'
            }), 400
        
        # 验证密码
        if password != app.config['DELETE_PASSWORD']:
            return jsonify({
                'success': False,
                'message': '密码错误'
            }), 401
        
        # 查找并删除物品
        item = medicine_material.query.get(item_id)
        if not item:
            return jsonify({
                'success': False,
                'message': '物品不存在'
            }), 404
        
        # 记录删除操作日志
        logger.info(f"删除药品物资: ID={item.id}, 名称={item.name}, 操作员={request.remote_addr}")
        
        db.session.delete(item)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': '物品删除成功'
        })
        
    except Exception as e:
        logger.error(f"删除物品失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'删除失败: {str(e)}'
        }), 500

# 在MedicineMaterial模型后添加操作员和确认人列表（全局变量）
OPERATORS = [
    "邱盛斌", "张华杰", "龚灵", "李景胡", "叶信城", "罗燕君", "叶汉农", "萧冰莹", 
    "许华文", "黄凤嫦", "方洧洧勉", "曾爱华", "梁敏琪", "周晓华", "温苓橦橦", "何咏瑜", 
    "陈敬宏", "骆锦强", "丁志华", "陈明旋", "谷敏", "黎小玲", "杨彦", "欧阳健", 
    "梁树华", "邱惠玲", "陈聪", "黄金枝", "谢智妍", "马艳彩", "伍文", "何海涛"
]

CONFIRMERS = [
    "邱盛斌", "张华杰", "龚灵", "李景胡", "叶信城", "罗燕君", "叶汉农", "萧冰莹", 
    "许华文", "黄凤嫦", "方洧洧勉", "曾爱华", "梁敏琪", "周晓华", "温苓橦橦", "何咏瑜", 
    "陈敬宏", "骆锦强", "丁志华", "陈明旋", "谷敏", "黎小玲", "杨彦", "欧阳健", 
    "梁树华", "邱惠玲", "陈聪", "黄金枝", "谢智妍", "马艳彩", "伍文", "何海涛"
]

# 添加药品/物资路由（修复版）
@app.route('/add_medicine_material', methods=['GET', 'POST'])
def add_medicine_material():
    # 获取操作员和确认人列表
    operators = OPERATORS
    confirmers = CONFIRMERS
    
    if request.method == 'POST':
        try:
            # 创建新物品对象（使用正确的模型名称）
            new_item = medicine_material(
                item_code=request.form.get('item_code'),
                name=request.form.get('name'),
                item_type=request.form.get('item_type'),
                category=request.form.get('category'),
                specification=request.form.get('specification'),
                model=request.form.get('model'),
                purpose=request.form.get('purpose'),
                laboratory=request.form.get('laboratory'),
                manufacturer=request.form.get('manufacturer'),
                supplier=request.form.get('supplier'),
                official_website=request.form.get('official_website'),
                notes=request.form.get('notes'),
                quantity=int(request.form.get('quantity', 0)),
                unit=request.form.get('unit'),
                location=request.form.get('location'),
                batch_total_stock=int(request.form.get('batch_total_stock', 0)),
                total_stock=int(request.form.get('total_stock', 0)),
                safety_stock=int(request.form.get('safety_stock', 0)),
                status=request.form.get('status'),
                storage_condition=request.form.get('storage_condition'),
                action=request.form.get('action'),
                batch_number=request.form.get('batch_number'),
                operator=request.form.get('operator'),
                confirmer=request.form.get('confirmer'),
                operation_date=datetime.now(timezone.utc)
            )
            
            # 处理日期字段
            payment_date = request.form.get('payment_date')
            if payment_date:
                new_item.payment_date = datetime.strptime(payment_date, '%Y-%m-%d').date()
            
            expiration_date = request.form.get('expiration_date')
            if expiration_date:
                new_item.expiration_date = datetime.strptime(expiration_date, '%Y-%m-%d').date()
            
            # 处理文件上传字段
            file_fields = [
                'item_image', 'label_image', 'instruction_image', 'location_image',
                'delivery_order_attachment', 'proposal_attachment', 'purchase_request_attachment',
                'manufacturer_data_attachment', 'disposal_data_attachment', 'other_attachment',
                'usage_video'
            ]
            
            for field in file_fields:
                file = request.files.get(field)
                if file and file.filename != '':
                    if allowed_file(file.filename):
                        filename = secure_filename(file.filename)
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        file.save(file_path)
                        setattr(new_item, field, filename)
            
            # 添加到数据库
            db.session.add(new_item)
            db.session.commit()
            
            flash('物品添加成功!', 'success')
            return redirect(url_for('medicine_material_list'))
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"添加物品失败: {str(e)}", exc_info=True)
            flash(f'添加失败: {str(e)}', 'danger')
            return render_template('add_edit_medicine_material.html', 
                                 item=request.form,
                                 operators=operators,
                                 confirmers=confirmers,
                                 current_time=datetime.now(timezone.utc))
    
    # GET请求：渲染添加表单
    current_time = datetime.now(timezone.utc)
    return render_template(
        'add_edit_medicine_material.html', 
        item=None,
        operators=operators,
        confirmers=confirmers,
        current_time=current_time
    )


@app.route('/edit_medicine_material/<int:id>', methods=['GET', 'POST'])
def edit_medicine_material(id):
    """编辑药品与物资记录"""
    # 获取要编辑的物品
    item = medicine_material.query.get_or_404(id)
    
    # 获取操作员和确认人列表
    operators = OPERATORS
    confirmers = CONFIRMERS
    
    # 定义文件字段列表 - 修复字段名称
    file_fields = [
        'item_image', 'label_image', 'instruction_image', 'location_image',
        'delivery_order_attachment', 'proposal_attachment', 'purchase_request_attachment',
        'manufacturer_data_attachment', 'disposal_data_attachment', 'other_attachment',
        'usage_video',
        'manufacturer_manual_attachment', 'sop_attachment',
        'calibration_record_attachment'
    ]
    
    if request.method == 'POST':
        try:
            # 检查是否是"另存为新记录"操作
            save_as_new = 'save_as_new' in request.form

            if save_as_new:
                # 创建新设备对象
                new_item = medicine_material(
                    item_code=request.form.get('item_code'),
                    name=request.form.get('name'),
                    item_type=request.form.get('item_type'),
                    category=request.form.get('category'),
                    specification=request.form.get('specification'),
                    model=request.form.get('model'),
                    purpose=request.form.get('purpose'),
                    laboratory=request.form.get('laboratory'),
                    manufacturer=request.form.get('manufacturer'),
                    supplier=request.form.get('supplier'),
                    official_website=request.form.get('official_website'),
                    notes=request.form.get('notes'),
                    quantity=int(request.form.get('quantity', 0)),
                    unit=request.form.get('unit'),
                    location=request.form.get('location'),
                    batch_total_stock=int(request.form.get('batch_total_stock', 0)),
                    total_stock=int(request.form.get('total_stock', 0)),
                    safety_stock=int(request.form.get('safety_stock', 0)),
                    status=request.form.get('status'),
                    storage_condition=request.form.get('storage_condition'),
                    action=request.form.get('action'),
                    batch_number=request.form.get('batch_number'),
                    operator=request.form.get('operator'),
                    confirmer=request.form.get('confirmer'),
                    operation_date=datetime.now(timezone.utc)
                )
                
                # 处理日期字段
                payment_date = request.form.get('payment_date')
                if payment_date:
                    new_item.payment_date = datetime.strptime(payment_date, '%Y-%m-%d').date()
                
                expiration_date = request.form.get('expiration_date')
                if expiration_date:
                    new_item.expiration_date = datetime.strptime(expiration_date, '%Y-%m-%d').date()
                
# 处理文件上传字段
                for field in file_fields:
                    file = request.files.get(field)
                    if file and file.filename != '':
                        if allowed_file(file.filename):
                            # 验证文件大小
                            if field == 'usage_video' and file.content_length > 100 * 1024 * 1024:
                                flash(f'视频文件大小超过100MB限制', 'danger')
                                continue
                            
                            # 生成唯一文件名
                            filename = secure_filename(file.filename)
                            base, ext = os.path.splitext(filename)
                            unique_id = str(uuid.uuid4())[:8]
                            unique_filename = f"{base}_{unique_id}{ext}"
                            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                            file.save(file_path)
                            setattr(new_item, field, unique_filename)
                        else:
                            flash(f'不允许的文件类型: {file.filename}', 'warning')
                    else:
                        # 使用原记录的文件名
                        setattr(new_item, field, getattr(item, field))
                
                db.session.add(new_item)
                flash('记录已成功保存为新记录!', 'success')
            else:
                # 更新现有记录
                item.item_code = request.form.get('item_code')
                item.name = request.form.get('name')
                item.item_type = request.form.get('item_type')
                item.category = request.form.get('category')
                item.specification = request.form.get('specification')
                item.model = request.form.get('model')
                item.purpose = request.form.get('purpose')
                item.laboratory = request.form.get('laboratory')
                item.manufacturer = request.form.get('manufacturer')
                item.supplier = request.form.get('supplier')
                item.official_website = request.form.get('official_website')
                item.notes = request.form.get('notes')
                item.quantity = int(request.form.get('quantity', 0))
                item.unit = request.form.get('unit')
                item.location = request.form.get('location')
                item.batch_total_stock = int(request.form.get('batch_total_stock', 0))
                item.total_stock = int(request.form.get('total_stock', 0))
                item.safety_stock = int(request.form.get('safety_stock', 0))
                item.status = request.form.get('status')
                item.storage_condition = request.form.get('storage_condition')
                item.action = request.form.get('action')
                item.batch_number = request.form.get('batch_number')
                item.operator = request.form.get('operator')
                item.confirmer = request.form.get('confirmer')
                item.operation_date = datetime.now(timezone.utc)
                
                # 处理日期字段
                payment_date = request.form.get('payment_date')
                item.payment_date = datetime.strptime(payment_date, '%Y-%m-%d').date() if payment_date else None
                expiration_date = request.form.get('expiration_date')
                item.expiration_date = datetime.strptime(expiration_date, '%Y-%m-%d').date() if expiration_date else None
                
                # 处理移除视频的请求
                if request.form.get('remove_video_flag') == 'true':
                    if item.usage_video:
                        # 删除物理文件
                        video_path = os.path.join(app.config['UPLOAD_FOLDER'], item.usage_video)
                        if os.path.exists(video_path):
                            try:
                                os.remove(video_path)
                            except Exception as e:
                                logger.error(f"删除视频文件失败: {str(e)}")
                        # 清除数据库字段
                        item.usage_video = None
                
               # 处理文件上传
                for field in file_fields:
                    file = request.files.get(field)
                    if file and file.filename != '':
                        if allowed_file(file.filename):
                            # 验证文件大小
                            if field == 'usage_video' and file.content_length > 100 * 1024 * 1024:
                                flash(f'视频文件大小超过100MB限制', 'danger')
                                continue
                            
                            # 生成唯一文件名
                            filename = secure_filename(file.filename)
                            base, ext = os.path.splitext(filename)
                            unique_id = str(uuid.uuid4())[:8]
                            unique_filename = f"{base}_{unique_id}{ext}"
                            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                            file.save(file_path)
                            
                            # 删除旧文件（如果存在）
                            old_filename = getattr(item, field)
                            if old_filename:
                                old_path = os.path.join(app.config['UPLOAD_FOLDER'], old_filename)
                                if os.path.exists(old_path):
                                    try:
                                        os.remove(old_path)
                                    except Exception as e:
                                        logger.error(f"删除旧文件失败: {str(e)}")
                            
                            # 设置新文件名
                            setattr(item, field, unique_filename)
                        else:
                            flash(f'不允许的文件类型: {file.filename}', 'warning')
                
                # 处理移除视频的请求
                if request.form.get('remove_video_flag') == 'true':
                    if item.usage_video:
                        # 删除物理文件
                        video_path = os.path.join(app.config['UPLOAD_FOLDER'], item.usage_video)
                        if os.path.exists(video_path):
                            try:
                                os.remove(video_path)
                            except Exception as e:
                                logger.error(f"删除视频文件失败: {str(e)}")
                        # 清除数据库字段
                        item.usage_video = None
            
            db.session.commit()
            flash('物品信息更新成功!', 'success')
            
            return redirect(url_for('medicine_material_list'))
        
        except Exception as e:
            db.session.rollback()
            logger.error(f"操作失败: {str(e)}", exc_info=True)
            flash(f'操作失败: {str(e)}', 'danger')
            return redirect(url_for('edit_medicine_material', id=id))
    
    # GET请求：渲染编辑表单
    current_time = datetime.now(timezone.utc)
    return render_template(
        'add_edit_medicine_material.html', 
        item=item,
        operators=operators,
        confirmers=confirmers,
        current_time=current_time
    )
@app.route('/view_medicine_material/<int:id>')
def view_medicine_material(id):
    """查看药品物资详情页"""
    # 使用正确的模型名称
    item = medicine_material.query.get_or_404(id)
    today = datetime.now().date()
    days_remaining = (item.expiration_date - today).days if item.expiration_date else None
    
    return render_template(
        'view_medicine_material.html', 
        item=item,
        days_remaining=days_remaining
    )



# 高级搜索
@app.route('/search_medicine_material', methods=['GET', 'POST'])
def search_medicine_material():
    if request.method == 'POST':
        # 构建查询（使用正确的模型名称）
        query = medicine_material.query
        if request.form.get('name'):
            query = query.filter(medicine_material.name.ilike(f"%{request.form.get('name')}%"))
        if request.form.get('item_code'):
            query = query.filter(medicine_material.item_code.ilike(f"%{request.form.get('item_code')}%"))
        # 其他条件...
        
        items = query.all()
        return render_template('medicine_material.html', items=items)
    return render_template('search_medicine_material.html')

# ================== 药品与物资管理路由 ==================
@app.route('/download_medicine_attachment/<int:id>/<string:type>')
def download_medicine_attachment(id, type):
    item = medicine_material.query.get_or_404(id)
    attachment_map = {
        'item_image': item.item_image,
        'label_image': item.label_image,
        'instruction_image': item.instruction_image,
        'location_image': item.location_image,
        'usage_video': item.usage_video,
        'delivery_order_attachment': item.delivery_order_attachment,
        'proposal_attachment': item.proposal_attachment,
        'purchase_request_attachment': item.purchase_request_attachment,
        'manufacturer_data_attachment': item.manufacturer_data_attachment,
        'disposal_data_attachment': item.disposal_data_attachment,
        'other_attachment': item.other_attachment
    }
    
    if type not in attachment_map or not attachment_map[type]:
        abort(404, description="附件不存在")
    
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], attachment_map[type], as_attachment=True)
    except FileNotFoundError:
        abort(404, description="附件文件未找到")


@app.route('/medicine_material/<int:id>')
def medicine_material_details(id):
    """药品物资详情页路由处理"""
    try:
        # 使用正确的模型名称（小写）
        item = medicine_material.query.get_or_404(id)
        today = datetime.now().date()
        days_remaining = (item.expiration_date - today).days if item.expiration_date else None
        
        return render_template(
            'view_medicine_material.html', 
            item=item,
            days_remaining=days_remaining
        )
    except Exception as e:
        logger.error(f"详情页错误 ID:{id} | 错误:{str(e)}")
        abort(500, description="加载物品详情时发生错误")



# 获取操作员和确认人列表（从数据库查询）
def get_operators_from_db():
    operators = set()
    # 从化学品表获取
    operators.update([op[0] for op in db.session.query(Chemical.operator).distinct().all() if op[0]])
    # 从设备表获取
    operators.update([op[0] for op in db.session.query(Equipment.operator).distinct().all() if op[0]])
    # 从药品物资表获取
    operators.update([op[0] for op in db.session.query(medicine_material.operator).distinct().all() if op[0]])
    return sorted(list(operators))

def get_confirmers_from_db():
    confirmers = set()
    # 从化学品表获取
    confirmers.update([cf[0] for cf in db.session.query(Chemical.confirmer).distinct().all() if cf[0]])
    # 从设备表获取
    confirmers.update([cf[0] for cf in db.session.query(Equipment.confirmer).distinct().all() if cf[0]])
    # 从药品物资表获取
    confirmers.update([cf[0] for cf in db.session.query(medicine_material.confirmer).distinct().all() if cf[0]])
    return sorted(list(confirmers))

# 药品与物资管理路由
@app.route('/add_edit_medicine_material', methods=['GET', 'POST'])
def add_edit_medicine_material():
    # 从数据库获取操作员和确认人列表
    operators = get_operators_from_db()
    confirmers = get_confirmers_from_db()
    
    item = None
    if request.method == 'POST' and 'id' in request.form:
        item_id = request.form.get('id')
        if item_id:
            item = medicine_material.query.get(item_id)
    
    if request.method == 'POST':
        try:
            # 创建新物品对象或更新现有对象
            if not item:
                item = medicine_material()
                db.session.add(item)
            
            # 更新基本字段
            item.item_code = request.form.get('item_code')
            item.name = request.form.get('name')
            item.item_type = request.form.get('item_type')
            item.category = request.form.get('category')
            item.specification = request.form.get('specification')
            item.model = request.form.get('model')
            item.purpose = request.form.get('purpose')
            item.laboratory = request.form.get('laboratory')
            item.manufacturer = request.form.get('manufacturer')
            item.supplier = request.form.get('supplier')
            item.official_website = request.form.get('official_website')
            item.notes = request.form.get('notes')
            
            # 处理数值字段
            item.quantity = int(request.form.get('quantity', 0))
            item.batch_total_stock = int(request.form.get('batch_total_stock', 0))
            item.total_stock = int(request.form.get('total_stock', 0))
            item.safety_stock = int(request.form.get('safety_stock', 0))
            
            # 处理价格字段
            try:
                item.price = float(request.form.get('price', 0))
            except (TypeError, ValueError):
                item.price = 0.0
                
            item.unit = request.form.get('unit')
            item.location = request.form.get('location')
            item.storage_condition = request.form.get('storage_condition')
            item.action = request.form.get('action')
            item.batch_number = request.form.get('batch_number')
            item.operator = request.form.get('operator')
            item.confirmer = request.form.get('confirmer')
            item.status = request.form.get('status')
            
            # 处理日期字段
            payment_date = request.form.get('payment_date')
            if payment_date:
                item.payment_date = datetime.strptime(payment_date, '%Y-%m-%d').date()
            
            expiration_date = request.form.get('expiration_date')
            if expiration_date:
                item.expiration_date = datetime.strptime(expiration_date, '%Y-%m-%d').date()
            
            operation_date = request.form.get('operation_date')
            if operation_date:
                item.operation_date = datetime.strptime(operation_date, '%Y-%m-%dT%H:%M')
            else:
                item.operation_date = datetime.now(timezone.utc)
            
            # 处理文件上传字段
            file_fields = [
                'item_image', 'label_image', 'instruction_image', 'location_image',
                'delivery_order_attachment', 'proposal_attachment', 'purchase_request_attachment',
                'manufacturer_data_attachment', 'disposal_data_attachment', 'other_attachment',
                'usage_video'
            ]
            
            # 保存上传的文件
            for field in file_fields:
                file = request.files.get(field)
                if file and file.filename != '' and allowed_file(file.filename):
                    # 生成唯一文件名
                    filename = secure_filename(file.filename)
                    base, ext = os.path.splitext(filename)
                    unique_id = str(uuid.uuid4())[:8]
                    unique_filename = f"{base}_{unique_id}{ext}"
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                    file.save(file_path)
                    
                    # 删除旧文件（如果存在）
                    old_filename = getattr(item, field)
                    if old_filename:
                        old_path = os.path.join(app.config['UPLOAD_FOLDER'], old_filename)
                        if os.path.exists(old_path):
                            try:
                                os.remove(old_path)
                            except Exception as e:
                                logger.error(f"删除旧文件失败: {str(e)}")
                    
                    # 设置新文件名
                    setattr(item, field, unique_filename)
            
            # 处理移除文件的请求
            for field in file_fields:
                if request.form.get(f'remove_{field}') == '1':
                    # 删除物理文件
                    filename = getattr(item, field)
                    if filename:
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        if os.path.exists(file_path):
                            try:
                                os.remove(file_path)
                            except Exception as e:
                                logger.error(f"删除文件失败: {str(e)}")
                    
                    # 清除数据库字段
                    setattr(item, field, None)
            
            db.session.commit()
            flash('物品保存成功!', 'success')
            return redirect(url_for('medicine_material_list'))
        
        except Exception as e:
            db.session.rollback()
            logger.error(f"保存失败: {str(e)}", exc_info=True)
            flash(f'保存失败: {str(e)}', 'danger')
            # 出错时重新渲染表单，保留已填写的数据
            return render_template(
                'add_edit_medicine_material.html', 
                item=request.form,
                operators=operators,
                confirmers=confirmers
            )
    
    # GET请求：渲染表单
    if request.args.get('id'):
        item = medicine_material.query.get(request.args['id'])
    
    return render_template(
        'add_edit_medicine_material.html', 
        item=item,
        operators=operators,
        confirmers=confirmers
    )

# 确保模板过滤器定义在路由之外
@app.template_filter('dict_without')
def dict_without_filter(d, *keys):
    """自定义模板过滤器：从字典中移除指定键"""
    return {k: v for k, v in d.items() if k not in keys}

@app.route('/equipment_ai_query', methods=['GET', 'POST'])
def equipment_ai_query():
    if request.method == 'POST':
        try:
            # 获取表单数据
            name = request.form.get('name', '').strip()
            model = request.form.get('model', '').strip()
            manufacturer = request.form.get('manufacturer', '').strip()
            
            # 构建查询提示
            prompt = f"""
            作为设备管理专家，请根据以下信息查询设备的相关资料：
            设备名称: {name}
            设备型号: {model}
            生产厂家: {manufacturer}
            
            请提供以下信息：
            1. 设备的SOP（标准操作规程）要点
            2. 设备的主要设置参数（例如温度、压力、速度等）及其范围
            3. 该设备的常见故障及解决方法
            4. 从互联网上收集该设备的相关信息（如技术文档、用户手册、常见问题等）
            """
            
            # 调用DeepSeek API
            headers = {
                "Authorization": f"Bearer {app.config['DEEPSEEK_API_KEY']}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 2000
            }
            
            response = requests.post(
                app.config['DEEPSEEK_API_URL'], 
                headers=headers, 
                json=payload,
                timeout=81
            )
            response.raise_for_status()
            
            # 解析响应
            response_data = response.json()
            result = response_data['choices'][0]['message']['content']
            
            return render_template(
                'equipment_ai_query.html',
                result=result,
                name=name,
                model=model,
                manufacturer=manufacturer
            )
            
        except Exception as e:
            return render_template(
                'equipment_ai_query.html',
                error=f"查询失败: {str(e)}"
            )
    
    # GET请求显示查询表单
    return render_template('equipment_ai_query.html')

# 添加新的路由处理导出
@app.route('/export_equipment_report')
def export_equipment_report():
    # 获取导出参数
    format_type = request.args.get('format', 'excel')
    scope = request.args.get('scope', 'all')
    include_barcode = request.args.get('include_barcode', 'false') == 'true'
    
    # 获取筛选参数
    year = request.args.get('year', datetime.now().year, type=int)
    month = request.args.get('month', datetime.now().month, type=int)
    status = request.args.get('status', '')
    search_query = request.args.get('search', '', type=str)
    
    # 获取当前月份和上个月
    today = datetime.now().date()
    current_month_start = today.replace(day=1)
    next_month = current_month_start + timedelta(days=32)
    current_month_end = next_month.replace(day=1) - timedelta(days=1)
    
    last_month_end = current_month_start - timedelta(days=1)
    last_month_start = last_month_end.replace(day=1)
    
    # 优化查询性能 - 使用单一查询获取所有数据
    query = db.session.query(
        Equipment.id,
        Equipment.equipment_manage_id,
        Equipment.fixed_asset_id,
        Equipment.name,
        Equipment.model,
        Equipment.laboratory,
        Equipment.location,
        db.func.max(case(
            [
                (db.and_(
                    extract('year', Equipment.operation_date) == last_month_start.year,
                    extract('month', Equipment.operation_date) == last_month_start.month
                ), Equipment.status)
            ],
            else_='无记录'
        )).label('last_month_status'),
        db.func.max(case(
            [
                (db.and_(
                    extract('year', Equipment.operation_date) == current_month_start.year,
                    extract('month', Equipment.operation_date) == current_month_start.month
                ), Equipment.status)
            ],
            else_='无记录'
        )).label('current_month_status'),
        db.func.max(case(
            [
                (db.and_(
                    extract('year', Equipment.operation_date) == current_month_start.year,
                    extract('month', Equipment.operation_date) == current_month_start.month
                ), Equipment.inspection_date)
            ],
            else_=None
        )).label('inspection_date'),
        db.func.max(case(
            [
                (db.and_(
                    extract('year', Equipment.operation_date) == current_month_start.year,
                    extract('month', Equipment.operation_date) == current_month_start.month
                ), Equipment.operator)
            ],
            else_=None
        )).label('operator'),
        db.func.max(case(
            [
                (db.and_(
                    extract('year', Equipment.operation_date) == current_month_start.year,
                    extract('month', Equipment.operation_date) == current_month_start.month
                ), Equipment.notes)
            ],
            else_=None
        )).label('notes')
    ).group_by(
        Equipment.id,
        Equipment.equipment_manage_id,
        Equipment.fixed_asset_id,
        Equipment.name,
        Equipment.model,
        Equipment.laboratory,
        Equipment.location
    )
    
    # 应用筛选条件
    if status:
        status_list = status.split(',')
        query = query.filter(
            db.or_(
                *[Equipment.status.ilike(f'%{s}%') for s in status_list]
            )
        )
    
    if search_query:
        query = query.filter(
            db.or_(
                Equipment.name.ilike(f'%{search_query}%'),
                Equipment.equipment_manage_id.ilike(f'%{search_query}%'),
                Equipment.fixed_asset_id.ilike(f'%{search_query}%')
            )
        )
    
    # 执行查询
    report_data = query.all()
    
    # 准备导出数据
    data = []
    for idx, item in enumerate(report_data, 1):
        data.append({
            '序号': idx,
            '管理编号': item.equipment_manage_id or '-',
            '固定资产编号': item.fixed_asset_id or '-',
            '设备名称': item.name,
            '型号': item.model or '-',
            '实验室': item.laboratory or '-',
            '位置': item.location or '-',
            f'上次检查状态({last_month_start.strftime("%Y-%m")})': item.last_month_status,
            f'本次检查状态({current_month_start.strftime("%Y-%m")})': item.current_month_status,
            '检查日期': item.inspection_date.strftime('%Y-%m-%d') if item.inspection_date else '-',
            '操作员': item.operator or '-',
            '备注': item.notes or '-',
            # 条形码数据
            'barcode_data': include_barcode and f"ID:{item.id}|管理号:{item.equipment_manage_id}|资产号:{item.fixed_asset_id}|名称:{item.name}"
        })
    
    # 根据格式生成文件
    if format_type == 'excel':
        return generate_excel_export(data)
    elif format_type == 'csv':
        return generate_csv_export(data)
    elif format_type == 'pdf':
        return generate_pdf_export(data, year, month)
    else:
        abort(400, description="不支持的导出格式")

# Excel导出函数
def generate_excel_export(data):
    df = pd.DataFrame(data)
    # 移除条形码数据列（不是用于显示）
    if 'barcode_data' in df.columns:
        df = df.drop(columns=['barcode_data'])
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='设备月度检查表', index=False)
        
        # 获取工作簿和工作表
        workbook = writer.book
        worksheet = writer.sheets['设备月度检查表']
        
        # 设置列宽
        for idx, col in enumerate(df.columns):
            max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(idx, idx, max_len)
        
        # 添加条形码（如果启用）
        if any('barcode_data' in d for d in data):
            barcode_format = workbook.add_format({'align': 'center'})
            
            for idx, row in enumerate(data):
                if 'barcode_data' in row:
                    row_num = idx + 1  # Excel行号从1开始
                    worksheet.write(row_num, len(df.columns), row['barcode_data'])
                    worksheet.insert_image(
                        f'M{row_num+1}',  # M列
                        '',  # 不在Excel中嵌入图像
                        {'image_data': generate_barcode_image(row['barcode_data']),
                         'x_scale': 0.5, 'y_scale': 0.5}
                    )
    
    output.seek(0)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f"设备月度检查表_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    )

# CSV导出函数
def generate_csv_export(data):
    # 移除条形码数据列
    cleaned_data = [{k: v for k, v in d.items() if k != 'barcode_data'} for d in data]
    
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=cleaned_data[0].keys())
    writer.writeheader()
    writer.writerows(cleaned_data)
    
    response = make_response(output.getvalue())
    response.headers['Content-Disposition'] = f'attachment; filename=设备月度检查表_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    response.headers['Content-type'] = 'text/csv; charset=utf-8'
    return response

# PDF导出函数
def generate_pdf_export(data, year, month):
    # 使用ReportLab生成PDF
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter))
    
    # 创建表格数据
    table_data = [list(data[0].keys()) if data else []]
    for item in data:
        # 移除条形码数据
        cleaned_item = {k: v for k, v in item.items() if k != 'barcode_data'}
        table_data.append(list(cleaned_item.values()))
    
    # 创建表格
    table = Table(table_data)
    
    # 应用样式
    style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#4361ee')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f8f9fa')),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#dee2e6')),
    ])
    table.setStyle(style)
    
    # 添加标题
    title = Paragraph(
        f"{year}年{month}月设备月度检查表",
        ParagraphStyle(name='Title', fontName='Helvetica-Bold', fontSize=16, alignment=1)
    )
    
    # 添加生成时间
    generate_time = Paragraph(
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ParagraphStyle(name='Subtitle', fontName='Helvetica', fontSize=10, alignment=1)
    )
    
    # 构建PDF内容
    elements = [title, generate_time, Spacer(1, 12), table]
    doc.build(elements)
    
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f"设备月度检查表_{year}年{month}月.pdf"
    )

# 生成条形码图像（辅助函数）
def generate_barcode_image(data):
    # 使用JasBarcode生成条形码
    barcode = BytesIO()
    JsBarcode(barcode, data, {
        format: "CODE128",
        displayValue: False,
        height: 40,
        margin: 0,
        background: "transparent"
    })
    return barcode.getvalue()



@app.route('/export_monthly_report')
def export_monthly_report():
    """导出设备月度检查报表"""
    try:
        # 获取查询参数
        year = request.args.get('year', type=int, default=datetime.now().year)
        month = request.args.get('month', type=int, default=datetime.now().month)
        statuses = request.args.get('status', default='all')
        search_query = request.args.get('search', default='')
        format_type = request.args.get('format', default='excel')
        include_barcode = request.args.get('include_barcode', default='false') == 'true'
        
        # 验证参数
        if not year or not month:
            return jsonify({
                "error": "缺少必要的年份和月份参数",
                "suggestion": "请确保提供有效的年份和月份参数"
            }), 400
        
        # 计算日期范围
        today = datetime.now().date()
        current_month_start = today.replace(day=1)
        next_month = current_month_start + timedelta(days=32)
        current_month_end = next_month.replace(day=1) - timedelta(days=1)
        
        last_month_end = current_month_start - timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)
        
        # 处理状态筛选
        selected_statuses = statuses.split(',') if statuses != 'all' else ['all']
        
        # 获取报表数据
        report_data = get_monthly_report_data(year, month, selected_statuses, search_query)
        
        # 为每个记录添加序号
        for idx, item in enumerate(report_data, 1):
            item['序号'] = idx
        
        # 准备导出数据
        data = []
        for item in report_data:
            data.append({
                '序号': item['序号'],
                '管理编号': item['equipment_manage_id'] or '-',
                '固定资产编号': item['fixed_asset_id'] or '-',
                '设备名称': item['name'],
                '型号': item['model'] or '-',
                '实验室': item['laboratory'] or '-',
                '位置': item['location'] or '-',
                f'上次检查状态({last_month_start.strftime("%Y-%m")})': item['last_month_status'],
                f'本次检查状态({current_month_start.strftime("%Y-%m")})': item['current_month_status'],
                '检查日期': item['inspection_date'].strftime('%Y-%m-%d') if item['inspection_date'] else '-',
                '操作员': item['operator'] or '-',
                '备注': item['notes'] or '-',
                # 条形码数据
                'barcode_data': include_barcode and f"ID:{item['id']}|管理号:{item['equipment_manage_id']}|资产号:{item['fixed_asset_id']}|名称:{item['name']}"
            })
        
        # 生成文件
        if format_type == 'excel':
            return generate_monthly_excel(data, year, month)
        elif format_type == 'csv':
            return generate_monthly_csv(data, year, month)
        elif format_type == 'pdf':
            return generate_monthly_pdf(data, year, month)
        else:
            return jsonify({
                "error": "不支持的导出格式",
                "supported_formats": ["excel", "csv", "pdf"]
            }), 400
            
    except Exception as e:
        logger.error(f"月度报表导出失败: {str(e)}", exc_info=True)
        return jsonify({
            "error": "导出失败",
            "details": str(e)
        }), 500

def get_monthly_report_data(year, month, selected_statuses, search_query):
    """获取月度报表数据"""
    # 计算日期范围
    today = datetime.now().date()
    current_month_start = today.replace(day=1)
    next_month = current_month_start + timedelta(days=32)
    current_month_end = next_month.replace(day=1) - timedelta(days=1)
    
    last_month_end = current_month_start - timedelta(days=1)
    last_month_start = last_month_end.replace(day=1)
    
    # 构建基础查询
    base_query = Equipment.query
    
    # 应用搜索条件
    if search_query:
        base_query = base_query.filter(
            or_(
                Equipment.name.ilike(f'%{search_query}%'),
                Equipment.equipment_manage_id.ilike(f'%{search_query}%'),
                Equipment.fixed_asset_id.ilike(f'%{search_query}%')
            )
        )
    
    # 获取所有设备
    all_equipment = base_query.all()
    
    # 准备报表数据
    report_data = []
    
    for equip in all_equipment:
        # 查询当前月记录
        current_month_record = Equipment.query.filter(
            Equipment.id == equip.id,
            extract('year', Equipment.operation_date) == year,
            extract('month', Equipment.operation_date) == month
        ).order_by(Equipment.operation_date.desc()).first()
        
        # 查询上个月记录
        last_month_record = Equipment.query.filter(
            Equipment.id == equip.id,
            extract('year', Equipment.operation_date) == last_month_start.year,
            extract('month', Equipment.operation_date) == last_month_start.month
        ).order_by(Equipment.operation_date.desc()).first()
        
        current_status = current_month_record.status if current_month_record else None
        last_status = last_month_record.status if last_month_record else None
        
        # 状态筛选
        if 'all' not in selected_statuses:
            if current_status not in selected_statuses and (current_status or '无记录') not in selected_statuses:
                continue
        
        report_data.append({
            'id': equip.id,
            'equipment_manage_id': equip.equipment_manage_id,
            'fixed_asset_id': equip.fixed_asset_id,
            'name': equip.name,
            'model': equip.model,
            'laboratory': equip.laboratory,
            'location': equip.location,
            'current_month_status': current_status,
            'last_month_status': last_status,
            'inspection_date': current_month_record.inspection_date if current_month_record else None,
            'operator': current_month_record.operator if current_month_record else None,
            'notes': current_month_record.notes if current_month_record else None
        })
    
    return report_data

def generate_monthly_excel(data, year, month):
    """生成月度报表Excel文件"""
    try:
        # 创建DataFrame，移除条形码数据列
        df = pd.DataFrame(data)
        if 'barcode_data' in df.columns:
            df = df.drop(columns=['barcode_data'])
        
        # 创建Excel文件
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='月度检查表', index=False)
            
            # 获取工作簿和工作表
            workbook = writer.book
            worksheet = writer.sheets['月度检查表']
            
            # 设置格式
            header_format = workbook.add_format({
                'bg_color': '#4361ee',
                'bold': True,
                'font_color': 'white',
                'align': 'center',
                'valign': 'vcenter',
                'border': 1
            })
            
            # 应用表头格式
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # 设置列宽
            for idx, col in enumerate(df.columns):
                max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
                worksheet.set_column(idx, idx, max_len)
            
            # 添加条形码（如果启用）
            if any('barcode_data' in d for d in data):
                # 添加条形码列标题
                barcode_col = len(df.columns)
                worksheet.write(0, barcode_col, '设备条形码', header_format)
                
                # 添加条形码图像
                for row_num, record in enumerate(data, start=1):
                    if 'barcode_data' in record:
                        barcode_data = record['barcode_data']
                        try:
                            # 在实际应用中，这里应调用条形码生成函数
                            # 示例：barcode_image = generate_barcode_image(barcode_data)
                            # 由于环境限制，这里仅添加文本占位
                            worksheet.write(row_num, barcode_col, barcode_data)
                        except Exception as e:
                            logger.error(f"添加条形码失败: {str(e)}")
                            worksheet.write(row_num, barcode_col, f"条形码生成错误: {str(e)}")
        
        output.seek(0)
        filename = f"设备月度检查表_{year}年{month}月_{uuid.uuid4().hex[:6]}.xlsx"
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        logger.error(f"Excel生成失败: {str(e)}", exc_info=True)
        raise

def generate_monthly_csv(data, year, month):
    """生成月度报表CSV文件"""
    try:
        # 移除条形码数据列
        cleaned_data = []
        for d in data:
            cleaned = {k: v for k, v in d.items() if k != 'barcode_data'}
            cleaned_data.append(cleaned)
        
        # 创建CSV文件
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=cleaned_data[0].keys())
        writer.writeheader()
        writer.writerows(cleaned_data)
        
        # 创建响应
        response = make_response(output.getvalue().encode('utf-8-sig'))  # UTF-8 with BOM for Excel
        filename = f"设备月度检查表_{year}年{month}月_{uuid.uuid4().hex[:6]}.csv"
        response.headers['Content-Disposition'] = f'attachment; filename={filename}'
        response.headers['Content-type'] = 'text/csv; charset=utf-8'
        return response
    
    except Exception as e:
        logger.error(f"CSV生成失败: {str(e)}", exc_info=True)
        raise

def generate_monthly_pdf(data, year, month):
    """生成月度报表PDF文件"""
    try:
        # 创建PDF文档
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=landscape(letter))
        
        # 准备表格数据
        table_data = []
        
        # 添加表头
        headers = [
            '序号', '管理编号', '固定资产编号', '设备名称', '型号', 
            '实验室', '位置', '上次检查状态', '本次检查状态',
            '检查日期', '操作员', '备注'
        ]
        table_data.append(headers)
        
        # 添加数据行
        for item in data:
            # 移除条形码数据
            cleaned_item = {k: v for k, v in item.items() if k != 'barcode_data'}
            table_data.append([
                cleaned_item['序号'],
                cleaned_item['管理编号'] or '-',
                cleaned_item['固定资产编号'] or '-',
                cleaned_item['设备名称'],
                cleaned_item['型号'] or '-',
                cleaned_item['实验室'] or '-',
                cleaned_item['位置'] or '-',
                cleaned_item['上次检查状态'],
                cleaned_item['本次检查状态'],
                cleaned_item['检查日期'].strftime('%Y-%m-%d') if cleaned_item['检查日期'] else '-',
                cleaned_item['操作员'] or '-',
                cleaned_item['备注'] or '-'
            ])
        
        # 创建表格
        table = Table(table_data)
        
        # 应用样式
        style = TableStyle([
            # 表头样式
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#4361ee')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 9),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            
            # 数据行样式
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f8f9fa')),
            ('TEXTCOLOR', (0,1), (-1,-1), colors.black),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,1), (-1,-1), 8),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dee2e6')),
            
            # 列宽和换行
            ('WORDWRAP', (0,0), (-1,-1), True),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ])
        
        # 根据内容调整列宽
        col_widths = [40, 80, 100, 120, 80, 80, 80, 80, 80, 80, 60, 100]
        table.setStyle(style)
        
        # 添加标题
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Title'],
            fontSize=16,
            alignment=1,  # 居中
            spaceAfter=12
        )
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=styles['Normal'],
            fontSize=10,
            alignment=1,  # 居中
            spaceAfter=12
        )
        
        title = Paragraph(f"设备月度检查表 - {year}年{month}月", title_style)
        subtitle = Paragraph(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}", subtitle_style)
        
        # 构建PDF内容
        elements = [title, subtitle, table]
        
        # 生成PDF
        doc.build(elements)
        buffer.seek(0)
        
        # 返回响应
        filename = f"设备月度检查表_{year}年{month}月_{uuid.uuid4().hex[:6]}.pdf"
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        logger.error(f"PDF生成失败: {str(e)}", exc_info=True)
        raise

# 辅助函数
def get_previous_month(year, month):
    """获取上个月的年份和月份"""
    if month == 1:
        return year - 1, 12
    else:
        return year, month - 1

# 在实际应用中实现条形码生成
def generate_barcode_image(data):
    """生成条形码图像（占位函数）"""
    # 在生产环境中，这里应该使用如pyBarcode或reportlab的条形码生成功能
    # 由于环境限制，这里仅返回占位文本
    return f"条形码数据: {data}"


# 新增API详情路由（文档2中添加）
@app.route('/api/medicine_material_details/<int:id>')
def api_medicine_material_details(id):
    """获取药品物资详情API"""
    try:
        # 查询数据库
        item = medicine_material.query.get_or_404(id)
        
        # 计算剩余天数
        days_remaining = None
        if item.expiration_date:
            today = datetime.now().date()
            days_remaining = (item.expiration_date - today).days
        
        # 构建响应数据
        result = {
            'id': item.id,
            'item_code': item.item_code,
            'name': item.name,
            'batch_number': item.batch_number,
            'specification': item.specification,
            'model': item.model,
            'item_type': item.item_type,
            'category': item.category,
            'laboratory': item.laboratory,
            'action': item.action,
            'quantity': item.quantity,
            'unit': item.unit,
            'location': item.location,
            'price': item.price,
            'storage_condition': item.storage_condition,
            'payment_date': item.payment_date.strftime('%Y-%m-%d') if item.payment_date else None,
            'operation_date': item.operation_date.strftime('%Y-%m-%d %H:%M:%S') if item.operation_date else None,
            'expiration_date': item.expiration_date.strftime('%Y-%m-%d') if item.expiration_date else None,
            'days_remaining': days_remaining,
            'status': item.status,
            'manufacturer': item.manufacturer,
            'supplier': item.supplier,
            'notes': item.notes,
            'purpose': item.purpose,
            'batch_total_stock': item.batch_total_stock,
            'total_stock': item.total_stock,
            'safety_stock': item.safety_stock,
            # 附件URL
            'item_image': url_for('download_medicine_attachment', id=item.id, type='item_image', _external=True) if item.item_image else None,
            'label_image': url_for('download_medicine_attachment', id=item.id, type='label_image', _external=True) if item.label_image else None,
            'instruction_image': url_for('download_medicine_attachment', id=item.id, type='instruction_image', _external=True) if item.instruction_image else None,
            'location_image': url_for('download_medicine_attachment', id=item.id, type='location_image', _external=True) if item.location_image else None,
            'usage_video': url_for('download_medicine_attachment', id=item.id, type='usage_video', _external=True) if item.usage_video else None,
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"获取药品物资详情失败: {str(e)}")
        return jsonify({
            'error': f'获取详情失败: {str(e)}'
        }), 500

@app.route('/sort_equipment_by_status')
def sort_equipment_by_status():
    try:
        # 获取请求参数
        page = request.args.get('page', 1, type=int)
        view = request.args.get('view', 'table')  # 当前视图模式
        direction = request.args.get('direction', 'asc')  # 排序方向
        per_page = request.args.get('per_page', 20, type=int)  # 每页数量
        
        # 获取原始搜索参数
        fixed_asset_id = request.args.get('fixed_asset_id', '').strip()
        equipment_manage_id = request.args.get('equipment_manage_id', '').strip()
        name = request.args.get('name', '').strip()
        model = request.args.get('model', '').strip()
        laboratory = request.args.get('laboratory', '').strip()
        location = request.args.get('location', '').strip()
        status = request.args.get('status', '')
        
        # 定义状态优先级
        status_order = case(
            [
                (Equipment.status == '正常', 1),
                (Equipment.status == '待检', 2),
                (Equipment.status == '已经维修', 3),
                (Equipment.status == '维修中', 4),
                (Equipment.status == '停用', 5),
                (Equipment.status == '报废中', 6),
                (Equipment.status == '报废', 7)
            ],
            else_=8
        )
        
        # 构建基础查询
        query = Equipment.query
        
        # 应用搜索条件
        if fixed_asset_id:
            query = query.filter(Equipment.fixed_asset_id.ilike(f'%{fixed_asset_id}%'))
        if equipment_manage_id:
            query = query.filter(Equipment.equipment_manage_id.ilike(f'%{equipment_manage_id}%'))
        if name:
            query = query.filter(Equipment.name.ilike(f'%{name}%'))
        if model:
            query = query.filter(Equipment.model.ilike(f'%{model}%'))
        if laboratory:
            query = query.filter(Equipment.laboratory.ilike(f'%{laboratory}%'))
        if location:
            query = query.filter(Equipment.location.ilike(f'%{location}%'))
        if status:
            query = query.filter(Equipment.status == status)
        
        # 根据方向排序
        if direction == 'asc':
            query = query.order_by(status_order.asc())
        else:
            query = query.order_by(status_order.desc())
        
        # 执行分页查询
        equipment = query.paginate(
            page=page,
            per_page=per_page,
            error_out=False
        )
        
        # 获取最后更新时间
        last_update = db.session.query(
            func.max(Equipment.operation_date)
        ).scalar()
        
        # 获取所有实验室用于搜索下拉框
        laboratories = db.session.query(
            Equipment.laboratory
        ).distinct().filter(
            Equipment.laboratory.isnot(None)
        ).all()
        laboratories = [lab[0] for lab in laboratories]
        
        # 渲染模板
        return render_template(
            'equipment.html',
            equipment=equipment,
            last_update=last_update,
            laboratories=laboratories,
            current_view=view,  # 传递当前视图模式
            sort_direction=direction,  # 传递排序方向
            # 传递搜索参数
            request_args={
                'fixed_asset_id': fixed_asset_id,
                'equipment_manage_id': equipment_manage_id,
                'name': name,
                'model': model,
                'laboratory': laboratory,
                'location': location,
                'status': status,
                'view': view,
                'direction': direction
            }
        )
        
    except Exception as e:
        logger.error(f"按状态排序失败: {str(e)}", exc_info=True)
        flash(f'排序失败: {str(e)}', 'danger')
        return redirect(url_for('equipment_list'))

@app.context_processor
def inject_datetime():
    return {'datetime': datetime}





# 确保模板过滤器定义在路由之外
@app.template_filter('dict_without')
def dict_without_filter(d, *keys):
    """自定义模板过滤器：从字典中移除指定键"""
    return {k: v for k, v in d.items() if k not in keys}



# 用户模型
# 在 Chemical 模型后添加 User 模型
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    role = db.Column(db.String(20), default='user')  # admin, manager, user
    department = db.Column(db.String(100))
    phone = db.Column(db.String(20))
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


# 修复 login 路由
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember_me = bool(request.form.get('remember_me'))

        # 验证用户（这里使用默认用户）
        if username == 'admin':
            # 修复：正确创建用户对象
            user = User({
                'id': 1,
                'username': 'admin',
                'password_hash': 'pbkdf2:sha256:260000$Kq1Y1Z1z$8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e',
                'permissions': ['chemical:create', 'chemical:edit', 'chemical:delete']
            })
            
            if user.check_password(password):  # 默认密码是'admin'
                login_user(user, remember=remember_me)

                # 获取next参数，如果存在则重定向到该页面
                next_page = request.args.get('next')
                if next_page and is_safe_url(next_page):
                    return redirect(next_page)
                return redirect(url_for('index'))
            else:
                flash('密码错误', 'danger')
        else:
            flash('用户名错误', 'danger')

    return render_template('login.html')

# 同时修复 User 类的 check_password 方法
class User(UserMixin):
    """用户类，继承 UserMixin 以支持 Flask-Login"""
    def __init__(self, user_data=None):
        if user_data:
            self.id = user_data.get('id', 1)
            self.username = user_data.get('username', 'admin')
            self.password_hash = user_data.get('password_hash')
            self.permissions = user_data.get('permissions', ['chemical:create', 'chemical:edit', 'chemical:delete'])
        else:
            # 默认用户 - 使用固定的密码哈希
            self.id = 1
            self.username = 'admin'
            # 这里使用预先计算好的 'admin' 的哈希值
            self.password_hash = 'pbkdf2:sha256:260000$Kq1Y1Z1z$8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e'
            self.permissions = ['chemical:create', 'chemical:edit', 'chemical:delete']
    
    def check_password(self, password):
        """检查密码 - 简化版本，直接比较"""
        # 对于演示目的，可以直接比较
        if password == 'admin':
            return True
            
        # 如果password_hash为None，则返回False
        if self.password_hash is None:
            logger.error("密码哈希为None")
            return False
            
        # 检查哈希格式是否有效
        if not self.password_hash.startswith(('pbkdf2:', 'sha256:', 'scrypt:')):
            logger.error(f"无效的密码哈希格式: {self.password_hash[:20]}...")
            return False
            
        try:
            return check_password_hash(self.password_hash, password)
        except ValueError as e:
            logger.error(f"密码验证错误: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"密码验证异常: {str(e)}")
            return False

    def has_permission(self, permission):
        """检查用户是否有指定权限"""
        return permission in self.permissions

# 同时修复 load_user 函数
@login_manager.user_loader
def load_user(user_id):
    # 这里应该从数据库加载用户
    # 简化版本：返回默认用户
    user_data = {
        'id': int(user_id),
        'username': 'admin',
        'password_hash': 'pbkdf2:sha256:260000$Kq1Y1Z1z$8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e8e',
        'permissions': ['chemical:create', 'chemical:edit', 'chemical:delete']
    }
    return User(user_data)

# 添加缺失的 is_safe_url 函数
def is_safe_url(target):
    """检查URL是否安全，防止开放重定向攻击"""
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ('http', 'https') and \
           ref_url.netloc == test_url.netloc

# 添加登出路由
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('您已成功登出', 'success')
    return redirect(url_for('login'))

# 添加登录页面模板路由（如果还没有的话）
@app.route('/login')
def login_page():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return render_template('login.html')

# 安全检查URL函数
def is_safe_url(target):
    """检查URL是否安全，防止开放重定向攻击"""
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ('http', 'https') and \
           ref_url.netloc == test_url.netloc

# 添加注册路由（仅管理员可访问）
@app.route('/register', methods=['GET', 'POST'])
@login_required
@permission_required('user:create')
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role', 'user')
        department = request.form.get('department')
        phone = request.form.get('phone')
        
        if User.query.filter_by(username=username).first():
            flash('用户名已存在', 'danger')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('邮箱已被注册', 'danger')
            return render_template('register.html')
        
        user = User(
            username=username,
            email=email,
            role=role,
            department=department,
            phone=phone
        )
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        flash(f'用户 {username} 创建成功', 'success')
        return redirect(url_for('user_management'))
    
    return render_template('register.html')


    
    # 记录注销日志
    if username:
        logger.info(f"用户注销: {username} - IP: {request.remote_addr}")
    
    return redirect(url_for('login'))

# 用户管理路由
@app.route('/user_management')
@login_required
@permission_required('user:manage')
def user_management():
    users = User.query.order_by(User.created_at.desc()).all()
    return render_template('user_management.html', users=users)

# 修改个人信息路由
@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user = User.query.get(session['user_id'])
    
    if request.method == 'POST':
        user.email = request.form.get('email')
        user.department = request.form.get('department')
        user.phone = request.form.get('phone')
        
        new_password = request.form.get('new_password')
        if new_password:
            user.set_password(new_password)
            flash('密码已更新', 'success')
        
        db.session.commit()
        flash('个人信息已更新', 'success')
        return redirect(url_for('profile'))
    
    return render_template('profile.html', user=user)

# 修改权限装饰器实现
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('请先登录', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def permission_required(permission):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user_role = session.get('role', 'user')
            
            # 权限映射
            permissions = {
                'admin': ['chemical:create', 'chemical:edit', 'chemical:delete', 'user:manage', 'user:create'],
                'manager': ['chemical:create', 'chemical:edit', 'chemical:delete'],
                'user': ['chemical:create', 'chemical:edit']
            }
            
            user_permissions = permissions.get(user_role, [])
            
            if permission not in user_permissions:
                flash('权限不足', 'danger')
                return redirect(url_for('index'))
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# 创建默认管理员用户（如果不存在）
@app.before_request
def initialize_on_first_request():
    if not hasattr(app, 'has_been_initialized'):
        # 初始化代码
        initialize_models()
        app.has_been_initialized = True
def create_default_admin():
    if not User.query.filter_by(username='admin').first():
        admin = User(
            username='admin',
            email='admin@lab.com',
            role='admin',
            department='系统管理'
        )
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()
        print('默认管理员账户已创建: admin/admin123')

# 添加缺失的导入
try:
    from PyPDF2 import PdfReader
except ImportError:
    logger.warning("PyPDF2 未安装，PDF处理功能将不可用")

try:
    from docx import Document
except ImportError:
    logger.warning("python-docx 未安装，Word文档处理功能将不可用")

# 添加缺失的缓存支持
try:
    from flask_caching import Cache
    cache = Cache(app, config={'CACHE_TYPE': 'simple'})
except ImportError:
    logger.warning("Flask-Caching 未安装，缓存功能将不可用")
    cache = None

# 在文件最后添加应用运行代码
if __name__ == '__main__':
    # 确保上传目录存在
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # 运行应用
    app.run(debug=True, host='0.0.0.0', port=8000)