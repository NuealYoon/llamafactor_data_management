import gradio as gr
from app.database.database import SessionLocal
from app.services.training_service import TrainingService
from app.services.dataset_service import DatasetService


def create_training_ui():
    """Create Gradio UI for training management"""

    with gr.Tab("Training Management"):
        gr.Markdown("# 🚀 Training Management")
        gr.Markdown("Create and manage model training jobs")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Create Training Job")
                job_name = gr.Textbox(label="Job Name", placeholder="my_training_job")
                select_dataset_train = gr.Dropdown(
                    label="Select Dataset", choices=[], allow_custom_value=False
                )
                base_model_path = gr.Textbox(
                    label="Base Model Path",
                    placeholder="meta-llama/Llama-2-7b-hf",
                    value="meta-llama/Llama-2-7b-hf",
                )
                job_description = gr.Textbox(
                    label="Description (Optional)",
                    placeholder="Training job description",
                    lines=2,
                )

                gr.Markdown("#### Training Parameters")
                with gr.Row():
                    learning_rate = gr.Number(
                        label="Learning Rate", value=5e-5, precision=0
                    )
                    num_epochs = gr.Number(label="Epochs", value=3, precision=0)

                with gr.Row():
                    batch_size = gr.Number(label="Batch Size", value=4, precision=0)
                    grad_accum = gr.Number(
                        label="Gradient Accumulation", value=8, precision=0
                    )

                with gr.Row():
                    max_seq_len = gr.Number(
                        label="Max Sequence Length", value=512, precision=0
                    )
                    lora_rank = gr.Number(label="LoRA Rank", value=8, precision=0)

                with gr.Row():
                    lora_alpha = gr.Number(label="LoRA Alpha", value=16, precision=0)
                    lora_dropout = gr.Number(
                        label="LoRA Dropout", value=0.05, precision=2
                    )

                create_job_btn = gr.Button("Create Training Job", variant="primary")
                create_job_output = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=2):
                gr.Markdown("### Training Jobs")
                refresh_jobs_btn = gr.Button("🔄 Refresh")
                filter_status = gr.Dropdown(
                    choices=["all", "pending", "running", "completed", "failed"],
                    label="Filter by Status",
                    value="all",
                )
                jobs_df = gr.Dataframe(
                    headers=[
                        "ID",
                        "Name",
                        "Status",
                        "Dataset",
                        "Base Model",
                        "Created",
                    ],
                    label="Training Jobs",
                    interactive=False,
                )

        gr.Markdown("---")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Control Training Job")
                control_job_id = gr.Number(label="Job ID", precision=0)
                with gr.Row():
                    start_job_btn = gr.Button("▶️ Start", variant="primary")
                    stop_job_btn = gr.Button("⏸️ Stop", variant="stop")
                control_output = gr.Textbox(label="Status", interactive=False)

            with gr.Column():
                gr.Markdown("### Training Details")
                detail_job_id = gr.Number(label="Job ID", precision=0)
                view_details_btn = gr.Button("View Details")
                job_details = gr.JSON(label="Job Information")

        gr.Markdown("---")

        with gr.Row():
            gr.Markdown("### Training Logs")
            log_job_id = gr.Number(label="Job ID", precision=0)
            view_logs_btn = gr.Button("View Logs")
            logs_output = gr.Textbox(
                label="Logs", lines=15, interactive=False, max_lines=20
            )

        # Event handlers
        def refresh_datasets_for_training():
            db = SessionLocal()
            try:
                datasets = DatasetService.get_datasets(db)
                dataset_names = [ds["name"] for ds in datasets]
                return gr.Dropdown(choices=dataset_names)
            except:
                return gr.Dropdown(choices=[])
            finally:
                db.close()

        def create_job_handler(
            name,
            dataset_name,
            base_model,
            description,
            lr,
            epochs,
            batch,
            grad_acc,
            seq_len,
            rank,
            alpha,
            dropout,
        ):
            if not name or not dataset_name:
                return "Error: Job name and dataset are required"

            db = SessionLocal()
            try:
                # Get dataset ID
                datasets = DatasetService.get_datasets(db)
                dataset_id = None
                for ds in datasets:
                    if ds["name"] == dataset_name:
                        dataset_id = ds["id"]
                        break

                if not dataset_id:
                    return "Error: Dataset not found"

                result = TrainingService.create_training_job(
                    db=db,
                    name=name,
                    dataset_id=dataset_id,
                    base_model=base_model,
                    description=description or None,
                    learning_rate=float(lr),
                    num_epochs=int(epochs),
                    batch_size=int(batch),
                    gradient_accumulation_steps=int(grad_acc),
                    max_seq_length=int(seq_len),
                    lora_rank=int(rank),
                    lora_alpha=int(alpha),
                    lora_dropout=float(dropout),
                )
                return f"✅ Training job created: {result['name']} (ID: {result['id']})"
            except Exception as e:
                return f"❌ Error: {str(e)}"
            finally:
                db.close()

        def refresh_jobs_handler(status_filter):
            db = SessionLocal()
            try:
                status = None if status_filter == "all" else status_filter
                jobs = TrainingService.get_training_jobs(db, status=status)
                data = [
                    [
                        job["id"],
                        job["name"],
                        job["status"],
                        job["dataset_id"],
                        job["base_model"],
                        job["created_at"],
                    ]
                    for job in jobs
                ]
                return data
            except Exception as e:
                return []
            finally:
                db.close()

        def start_job_handler(job_id):
            if not job_id:
                return "Error: Job ID is required"

            db = SessionLocal()
            try:
                result = TrainingService.start_training(db, int(job_id))
                return f"✅ {result['message']}"
            except Exception as e:
                return f"❌ Error: {str(e)}"
            finally:
                db.close()

        def stop_job_handler(job_id):
            if not job_id:
                return "Error: Job ID is required"

            db = SessionLocal()
            try:
                result = TrainingService.stop_training(db, int(job_id))
                return f"✅ Training job stopped (Status: {result['status']})"
            except Exception as e:
                return f"❌ Error: {str(e)}"
            finally:
                db.close()

        def view_details_handler(job_id):
            if not job_id:
                return {}

            db = SessionLocal()
            try:
                from app.database import crud

                job = crud.get_training_job(db, int(job_id))
                if not job:
                    return {"error": "Job not found"}

                return {
                    "id": job.id,
                    "name": job.name,
                    "status": job.status.value,
                    "dataset_id": job.dataset_id,
                    "base_model": job.base_model,
                    "learning_rate": job.learning_rate,
                    "num_epochs": job.num_epochs,
                    "batch_size": job.batch_size,
                    "created_at": job.created_at.isoformat() if job.created_at else None,
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "completed_at": job.completed_at.isoformat()
                    if job.completed_at
                    else None,
                }
            except Exception as e:
                return {"error": str(e)}
            finally:
                db.close()

        def view_logs_handler(job_id):
            if not job_id:
                return "Please enter a Job ID"

            db = SessionLocal()
            try:
                logs = TrainingService.get_training_logs(db, int(job_id))
                return "\n".join(logs)
            except Exception as e:
                return f"Error: {str(e)}"
            finally:
                db.close()

        # Connect events
        gr.on(
            triggers=[create_job_btn.click],
            fn=refresh_datasets_for_training,
            outputs=select_dataset_train,
        ).then(
            fn=create_job_handler,
            inputs=[
                job_name,
                select_dataset_train,
                base_model_path,
                job_description,
                learning_rate,
                num_epochs,
                batch_size,
                grad_accum,
                max_seq_len,
                lora_rank,
                lora_alpha,
                lora_dropout,
            ],
            outputs=create_job_output,
        )

        refresh_jobs_btn.click(
            refresh_jobs_handler, inputs=filter_status, outputs=jobs_df
        )

        filter_status.change(refresh_jobs_handler, inputs=filter_status, outputs=jobs_df)

        start_job_btn.click(
            start_job_handler, inputs=control_job_id, outputs=control_output
        )

        stop_job_btn.click(
            stop_job_handler, inputs=control_job_id, outputs=control_output
        )

        view_details_btn.click(
            view_details_handler, inputs=detail_job_id, outputs=job_details
        )

        view_logs_btn.click(view_logs_handler, inputs=log_job_id, outputs=logs_output)

    return None
