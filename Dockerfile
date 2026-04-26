# syntax=docker/dockerfile:1.6                                                                                     
                                    
FROM python:3.11-slim AS builder                                                                                   
                                    
RUN python -m venv /opt/venv                           
ENV PATH="/opt/venv/bin:$PATH"      
                                                                                                                    
RUN pip install --no-cache-dir --upgrade pip
                                                                                                                    
COPY requirements.txt .
                                                        
RUN pip install --no-cache-dir -r requirements.txt                                                                 

                                                                                                                    
FROM python:3.11-slim
                                                        
WORKDIR /app                        

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
                                                                                                                    
COPY app ./app
                                                                                                                    
ENV PYTHONUNBUFFERED=1
EXPOSE 8000                                            
                                    
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]   