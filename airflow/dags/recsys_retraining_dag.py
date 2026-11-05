from datetime import datetime, timedelta
import os
import sys

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    DAG = object

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2)
}

def task_spark_preprocess(**kwargs):
    from src.spark_pipeline import run_pyspark_pipeline
    metadata = run_pyspark_pipeline()
    return f"Processed {metadata['train_samples']} train samples."

def task_train_retrieval(**kwargs):
    from src.retrieval.train_retrieval import train_retrieval_model
    train_retrieval_model(epochs=6, batch_size=256)
    return "Two-tower retrieval training finished."

def task_update_faiss(**kwargs):
    from src.retrieval.faiss_index import build_faiss_index
    index = build_faiss_index()
    return f"Indexed {index.ntotal} candidate items."

def task_train_rankers(**kwargs):
    from src.ranking.train_ranking import train_ranking_models
    summary = train_ranking_models()
    return f"Trained rankers. DCN AUC: {summary['DCN_v2']['best_auc']:.4f}"

def task_evaluate_gatekeep(**kwargs):
    from src.evaluation.offline_eval import run_offline_evaluation
    benchmark = run_offline_evaluation(sample_users=50)
    ndcg = benchmark["Stage2_DCN_v2"]["ndcg@10"]
    return "Evaluation quality gate passed."

def task_export_triton(**kwargs):
    from src.serving.triton_export import export_all_models_to_triton
    export_all_models_to_triton()
    return "Triton ONNX models successfully exported."

def task_invalidate_cache(**kwargs):
    from src.cache.redis_cache import RecommendationCache
    cache = RecommendationCache()
    cache.clear_all()
    return "Redis recommendation cache cleared."

if AIRFLOW_AVAILABLE:
    dag = DAG(
        'recommendation_engine_retraining_pipeline',
        default_args=default_args,
        schedule_interval='0 2 * * *',
        catchup=False,
        tags=['recommendation', 'retrieval', 'ranking', 'faiss', 'triton']
    )

    t1_spark = PythonOperator(
        task_id='extract_and_preprocess_spark',
        python_callable=task_spark_preprocess,
        dag=dag
    )

    t2_retrieval = PythonOperator(
        task_id='train_two_tower_retrieval',
        python_callable=task_train_retrieval,
        dag=dag
    )

    t3_faiss = PythonOperator(
        task_id='update_faiss_index',
        python_callable=task_update_faiss,
        dag=dag
    )

    t4_ranking = PythonOperator(
        task_id='train_ranking_models',
        python_callable=task_train_rankers,
        dag=dag
    )

    t5_gatekeep = PythonOperator(
        task_id='evaluate_and_gatekeep',
        python_callable=task_evaluate_gatekeep,
        dag=dag
    )

    t6_triton = PythonOperator(
        task_id='export_triton_onnx',
        python_callable=task_export_triton,
        dag=dag
    )

    t7_cache = PythonOperator(
        task_id='invalidate_redis_cache',
        python_callable=task_invalidate_cache,
        dag=dag
    )

    t1_spark >> t2_retrieval >> t3_faiss
    t1_spark >> t4_ranking
    [t3_faiss, t4_ranking] >> t5_gatekeep >> t6_triton >> t7_cache
